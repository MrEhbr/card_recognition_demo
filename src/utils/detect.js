import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

export const detectImage = async (image, canvas, session, topk, iouThreshold, scoreThreshold, inputShape) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new Tensor("float32", input.data32F, inputShape);
  const config = new Tensor("float32", new Float32Array([topk, iouThreshold, scoreThreshold]));
  const { output0 } = await session.net.run({ images: tensor });
  const { selected } = await session.nms.run({ detection: output0, config: config });

  const boxes = [];

  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]);
    const box = data.slice(0, 4);
    const scores = data.slice(4);
    const score = Math.max(...scores);
    const label = scores.indexOf(score);

    const [x, y, w, h] = [(box[0] - 0.5 * box[2]) * xRatio, (box[1] - 0.5 * box[3]) * yRatio, box[2] * xRatio, box[3] * yRatio];

    boxes.push({
      label: label,
      probability: score,
      bounding: [x, y, w, h],
    });
  }

  renderBoxes(canvas, boxes);
  input.delete();
};

const preprocessing = (source, modelWidth, modelHeight) => {
  const mat = cv.imread(source);
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3);
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);

  const maxSize = Math.max(matC3.rows, matC3.cols);
  const xPad = maxSize - matC3.cols,
    xRatio = maxSize / matC3.cols;
  const yPad = maxSize - matC3.rows,
    yRatio = maxSize / matC3.rows;
  const matPad = new cv.Mat();
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT);

  const input = cv.blobFromImage(matPad, 1 / 255.0, new cv.Size(modelWidth, modelHeight), new cv.Scalar(0, 0, 0), true, false);

  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

let stopFlag = false;

export const detectVideo = async (video, canvas, session, topk, iouThreshold, scoreThreshold, inputShape) => {
  stopFlag = false;

  const captureFrame = async () => {
    if (!video.paused && !video.ended && !stopFlag) {
      const [modelWidth, modelHeight] = inputShape.slice(2);

      const canvasForFrame = document.createElement("canvas");
      canvasForFrame.width = video.videoWidth;
      canvasForFrame.height = video.videoHeight;
      const context = canvasForFrame.getContext("2d");
      context.drawImage(video, 0, 0, canvasForFrame.width, canvasForFrame.height);

      const mat = cv.imread(canvasForFrame);
      const [input, xRatio, yRatio] = preprocessing(canvasForFrame, modelWidth, modelHeight);

      const tensor = new Tensor("float32", input.data32F, inputShape);
      const config = new Tensor("float32", new Float32Array([topk, iouThreshold, scoreThreshold]));
      const { output0 } = await session.net.run({ images: tensor });
      const { selected } = await session.nms.run({ detection: output0, config: config });

      const boxes = [];
      for (let idx = 0; idx < selected.dims[1]; idx++) {
        const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]);
        const box = data.slice(0, 4);
        const scores = data.slice(4);
        const score = Math.max(...scores);
        const label = scores.indexOf(score);

        const [x, y, w, h] = [(box[0] - 0.5 * box[2]) * xRatio, (box[1] - 0.5 * box[3]) * yRatio, box[2] * xRatio, box[3] * yRatio];

        boxes.push({
          label: label,
          probability: score,
          bounding: [x, y, w, h],
        });
      }

      renderBoxes(canvas, boxes);
      input.delete();
    }
    if (!stopFlag) {
      requestAnimationFrame(captureFrame);
    }
  };

  captureFrame();
};

export const stopVideoDetection = () => {
  stopFlag = true;
};
