package main

import (
	"fmt"
	"image"
	"time"

	"github.com/Danile71/go-face"
	"gocv.io/x/gocv"
)

const (
	netModel    = "models/face-detection-retail-0005.bin"
	netProto    = "models/face-detection-retail-0005.xml"
	cnnModel    = "models/mmod_human_face_detector.dat"
	shapeModel  = "models/shape_predictor_68_face_landmarks.dat"
	descrModel  = "models/dlib_face_recognition_resnet_model_v1.dat"
	ageModel    = "models/dnn_age_predictor_v1.dat"
	genderModel = "models/dnn_gender_classifier_v1.dat"
)

//openvino detecting
//dlib recognize

func main() {
	var (
		pt      = image.Pt(300, 300)
		scalar  = gocv.NewScalar(0, 0, 0, 0)
		faceNet = gocv.ReadNet(netModel, netProto)
	)

	faceNet.SetPreferableBackend(gocv.NetBackendDefault)
	faceNet.SetPreferableTarget(gocv.NetTargetCPU)
	defer faceNet.Close()

	rec, err := face.NewRecognizer()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer rec.Close()

	err = rec.SetShapeModel(shapeModel)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = rec.SetDescriptorModel(descrModel)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = rec.SetAgeModel(ageModel)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = rec.SetGenderModel(genderModel)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = rec.SetCNNModel(cnnModel)
	if err != nil {
		fmt.Println(err)
		return
	}
	start := time.Now()
	defer func() {
		fmt.Println("elapsed", time.Since(start))
	}()
	img := gocv.IMRead("test.jpg", gocv.IMReadUnchanged)
	defer img.Close()

	width := img.Cols()
	height := img.Rows()

	blob := gocv.BlobFromImage(img, 1, pt, scalar, false, false)
	defer blob.Close()

	faceNet.SetInput(blob, "")

	outputFace := faceNet.Forward("")
	defer outputFace.Close()

	for i := 0; i < outputFace.Total(); i += 7 {
		confidence := outputFace.GetFloatAt(0, i+2)

		if confidence > 0.5 {

			left := int(outputFace.GetFloatAt(0, i+3) * float32(width))
			top := int(outputFace.GetFloatAt(0, i+4) * float32(height))
			right := int(outputFace.GetFloatAt(0, i+5) * float32(width))
			bottom := int(outputFace.GetFloatAt(0, i+6) * float32(height))

			//bugfix
			if left < 0 {
				left = 0
			}
			if top < 0 {
				top = 0
			}
			if right > img.Cols() {
				right = img.Cols()
			}
			if bottom > img.Rows() {
				bottom = img.Rows()
			}

			r := image.Rect(left, top, right, bottom)
			submat := img.Region(r)

			submatgray := gocv.NewMat()
			gocv.CvtColor(submat, &submatgray, gocv.ColorBGRToGray)

			faces, err := rec.DetectFromMatCNN(submatgray)
			submatgray.Close()

			if err != nil {
				fmt.Println(err)
				continue
			}

			for _, f := range faces {
				rec.Recognize(&f)
				rec.GetAge(&f)
				rec.GetGender(&f)
				fmt.Println("detected:", f.Rectangle, " age:", f.Age, " gender:", f.Gender)
				f.Close()
			}
			submat.Close()
		}

	}
}
