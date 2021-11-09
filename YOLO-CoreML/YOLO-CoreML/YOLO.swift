import Foundation
import UIKit
import CoreML

class YOLO {
  public static let inputWidth = 608 //416 //608
  public static let inputHeight = 608 // 416 //608
  public static let maxBoundingBoxes = 10

  // Tweak these values to get more or fewer predictions.
  let confidenceThreshold: Float = 0.6
  let iouThreshold: Float = 0.213
  let isTiny = true
  let numClasses = 80
  let boxesPerCell = 3

  struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }

    let model =  Yolov4()
    let tinyModel = Yolov4_tiny()
    
  public init() { }

  public func predict(image: CVPixelBuffer) throws -> [Prediction] {
    if isTiny {
        if let output = try? tinyModel.prediction(input_1: image) {
            // The order of output Identity will be random, please double check of outputs order here
            return computeBoundingBoxes(features: [output.Identity, output.Identity_1])
        } else {
          return []
        }
    } else {
        if let output = try? model.prediction(input_1: image) {
            // The order of output Identity will be random, please double check of outputs order here
            return computeBoundingBoxes(features: [output.Identity_1, output.Identity, output.Identity_2])
        } else {
          return []
        }
    }
  }

  public func computeBoundingBoxes(features: [MLMultiArray]) -> [Prediction] {
    // Double check out features order here, should be in ascending order 
//    if !isTiny {
//        print("Features count: \(features[0].count) \(features[1].count) \(features[2].count)")
//    } else {
//        print("Features count: \(features[0].count) \(features[1].count)")
//    }

    var predictions = [Prediction]()

    let blockSize: Float = 32

    // For the 416x416 input size, image is divided into a 13x13 grid. Each of these grid
    // cells will predict 3 bounding boxes (boxesPerCell). A bounding box consists of
    // five data items: x, y, width, height, and a confidence score. Each grid
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 255x13x13 elements.

    // NOTE: It turns out that accessing the elements in the multi-array as
    // `features[[channel, cy, cx] as [NSNumber]].floatValue` is kinda slow.
    // It's much faster to use direct memory access to the features.
    let gridCellH = YOLO.inputHeight / 32
    let gridCellW = YOLO.inputWidth / 32
    let gridHeight = isTiny ? [gridCellH, gridCellH * 2] : [gridCellH, gridCellH * 2, gridCellH * 4]
    let gridWidth = isTiny ? [gridCellW, gridCellW * 2] : [gridCellW, gridCellW * 2, gridCellW * 4]
    
    var featurePointer = UnsafeMutablePointer<Float32>(OpaquePointer(features[0].dataPointer))

    let channelStride = 1
    var yStride = features[0].strides[1].intValue
    var xStride = features[0].strides[2].intValue

    func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
        let offset = channel*channelStride + y*yStride + x*xStride
        return offset
    }

    let featuresNum = isTiny ? 2 : 3
    for i in 0..<featuresNum {
        featurePointer = UnsafeMutablePointer<Float32>(OpaquePointer(features[i].dataPointer))
        yStride = features[i].strides[1].intValue
        xStride = features[i].strides[2].intValue

        for cy in 0..<gridHeight[i] {
            for cx in 0..<gridWidth[i] {
                for b in 0..<boxesPerCell {
                    // For the first bounding box (b=0) we have to read channels 0-24,
                    // for b=1 we have to read channels 25-49, and so on.
                    let channel = b*(numClasses + 5)
                    
                    // The fast way:
                    let tx = Float32(featurePointer[offset(channel    , cx, cy)])
                    let ty = Float32(featurePointer[offset(channel + 1, cx, cy)])
                    let tw = Float32(featurePointer[offset(channel + 2, cx, cy)])
                    let th = Float32(featurePointer[offset(channel + 3, cx, cy)])
                    let tc = Float32(featurePointer[offset(channel + 4, cx, cy)])
                    
                    // The predicted tx and ty coordinates are relative to the location
                    // of the grid cell; we use the logistic sigmoid to constrain these
                    // coordinates to the range 0 - 1. Then we add the cell coordinates
                    // (0-12) and multiply by the number of pixels per grid cell (32).
                    // Now x and y represent center of the bounding box in the original
                    // 416x416 image space.
                    let scale = powf(2.0,Float32(i)) // scale pos by 2^i where i is the scale pyramid level
                    let x = (Float32(cx) * blockSize * scale + sigmoid(tx))/scale
                    let y = (Float32(cy) * blockSize * scale + sigmoid(ty))/scale
                    
                    // The size of the bounding box, tw and th, is predicted relative to
                    // the size of an "anchor" box. Here we also transform the width and
                    // height into the original 416x416 image space.
                    let w = exp(tw) * scale * (isTiny ? anchorsTiny[i][2*b    ] : anchors[i][2*b    ])
                    let h = exp(th) * scale * (isTiny ? anchorsTiny[i][2*b + 1] : anchors[i][2*b + 1])
                    
                    // The confidence value for the bounding box is given by tc. We use
                    // the logistic sigmoid to turn this into a percentage.
                    let confidence = sigmoid(tc)
                    
                    // Gather the predicted classes for this anchor box and softmax them,
                    // so we can interpret these numbers as percentages.
                    var classes = [Float32](repeating: 0, count: numClasses)
                    for c in 0..<numClasses {
                        // The slow way:
                        //classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue
                        
                        // The fast way:
                        classes[c] = Float32(featurePointer[offset(channel + 5 + c, cx, cy)])
                    }
                    classes = softmax(classes)
                    
                    // Find the index of the class with the largest score.
                    let (detectedClass, bestClassScore) = classes.argmax()
                    
                    // Combine the confidence score for the bounding box, which tells us
                    // how likely it is that there is an object in this box (but not what
                    // kind of object it is), with the largest class prediction, which
                    // tells us what kind of object it detected (but not where).
                    let confidenceInClass = bestClassScore * confidence
                    
                    // Since we compute 13x13x3 = 507 bounding boxes, we only want to
                    // keep the ones whose combined score is over a certain threshold.
                    if confidenceInClass > confidenceThreshold {
                        let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                          width: CGFloat(w), height: CGFloat(h))
                        
                        let prediction = Prediction(classIndex: detectedClass,
                                                    score: confidenceInClass,
                                                    rect: rect)
                        predictions.append(prediction)
                    }
                }
            }
        }
    }

    // We already filtered out any bounding boxes that have very low scores,
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
    return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
  }
}
