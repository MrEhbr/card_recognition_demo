import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Box,
  Button,
  Image,
  Input,
  Select,
  Text,
  VStack,
  HStack,
  Heading,
  FormControl,
  FormLabel,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Grid,
  GridItem,
  Spinner,
  Center,
  RadioGroup,
  Radio,
  Stack,
} from "@chakra-ui/react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [modelChoice, setModelChoice] = useState("predefined");
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const inputModel = useRef(null);

  // Dynamic Configs
  const [config, setConfig] = useState({
    topk: 100,
    iouThreshold: 0.45,
    scoreThreshold: 0.25,
  });

  const [sliderConfig, setSliderConfig] = useState(config);
  const [isImageLoaded, setIsImageLoaded] = useState(false);

  const modelInputShape = useMemo(() => [1, 3, 640, 640], []);

  const handleSliderConfigChange = (name, value) => {
    setSliderConfig((prevConfig) => ({
      ...prevConfig,
      [name]: value,
    }));
  };

  const applySliderConfig = () => {
    setConfig(sliderConfig);
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
  };

  const handleModelChoiceChange = (value) => {
    setModelChoice(value);
    setModel("");
    setSession(null); // Reset session when switching model choice
  };

  const fetchModels = async () => {
    let modelFiles;
    const response = await fetch(`${process.env.PUBLIC_URL}/modelList.json`);
    modelFiles = await response.json();
    setModels(modelFiles);
    if (modelFiles.length > 0) setModel(modelFiles[0]);
  };

  const loadModel = useCallback(
    async (modelSource) => {
      setLoading({ text: `Loading model`, progress: null });
      let yolov8;
      let nms;

      if (modelSource instanceof File) {
        const reader = new FileReader();
        reader.onload = async (e) => {
          const buffer = e.target.result;
          yolov8 = await InferenceSession.create(buffer);
          const arrBufNMS = await download(`${process.env.PUBLIC_URL}/model/nms-yolov8.onnx`, ["Loading NMS model", setLoading]);
          nms = await InferenceSession.create(arrBufNMS);

          setLoading({ text: "Warming up model...", progress: null });
          const tensor = new Tensor("float32", new Float32Array(modelInputShape.reduce((a, b) => a * b)), modelInputShape);
          await yolov8.run({ images: tensor });

          setSession({ net: yolov8, nms: nms });
          setLoading(null);
          setModel("Custom Model");
        };
        reader.readAsArrayBuffer(modelSource);
      } else {
        const baseModelURL = `${process.env.PUBLIC_URL}/model`;
        const arrBufNet = await download(`${baseModelURL}/${modelSource}`, [`Loading ${modelSource}`, setLoading]);
        yolov8 = await InferenceSession.create(arrBufNet);
        const arrBufNMS = await download(`${baseModelURL}/nms-yolov8.onnx`, ["Loading NMS model", setLoading]);
        nms = await InferenceSession.create(arrBufNMS);

        setLoading({ text: "Warming up model...", progress: null });
        const tensor = new Tensor("float32", new Float32Array(modelInputShape.reduce((a, b) => a * b)), modelInputShape);
        await yolov8.run({ images: tensor });

        setSession({ net: yolov8, nms: nms });
        setLoading(null);
        setModel(modelSource);
      }
    },
    [modelInputShape],
  );

  useEffect(() => {
    cv["onRuntimeInitialized"] = () => {
      fetchModels();
    };
  }, []);

  const performDetection = useCallback(() => {
    if (image && session && isImageLoaded) {
      const img = imageRef.current;
      const canvas = canvasRef.current;
      detectImage(img, canvas, session, config.topk, config.iouThreshold, config.scoreThreshold, modelInputShape);
    }
  }, [image, session, config, isImageLoaded, modelInputShape]);

  useEffect(() => {
    if (model && modelChoice === "predefined") {
      loadModel(model);
    }
  }, [model, modelChoice, loadModel]);

  useEffect(() => {
    performDetection();
  }, [config, performDetection]);

  const handleImageLoad = () => {
    setIsImageLoaded(true);
    performDetection();
  };

  return (
    <Box p={4}>
      {loading && (
        <Center position="fixed" top="0" left="0" right="0" bottom="0" backgroundColor="rgba(255, 255, 255, 0.8)" zIndex="9999">
          <VStack spacing={4}>
            <Spinner size="xl" />
            <Text>{loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}</Text>
          </VStack>
        </Center>
      )}
      <Grid templateColumns="1fr 3fr" gap={4}>
        <GridItem borderRight="1px solid #ddd" paddingRight={4}>
          <VStack spacing={6} align="stretch">
            <Heading as="h2" size="lg" mb={4}>
              Settings
            </Heading>
            <FormControl>
              <FormLabel>Top K</FormLabel>
              <Slider
                name="topk"
                min={1}
                max={100}
                value={sliderConfig.topk}
                onChange={(val) => handleSliderConfigChange("topk", val)}
                onChangeEnd={applySliderConfig}
              >
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb />
              </Slider>
              <Text mt={2}>{sliderConfig.topk}</Text>
            </FormControl>
            <FormControl>
              <FormLabel>IoU Threshold</FormLabel>
              <Slider
                name="iouThreshold"
                min={0}
                max={1}
                step={0.01}
                value={sliderConfig.iouThreshold}
                onChange={(val) => handleSliderConfigChange("iouThreshold", val)}
                onChangeEnd={applySliderConfig}
              >
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb />
              </Slider>
              <Text mt={2}>{sliderConfig.iouThreshold}</Text>
            </FormControl>
            <FormControl>
              <FormLabel>Score Threshold</FormLabel>
              <Slider
                name="scoreThreshold"
                min={0}
                max={1}
                step={0.01}
                value={sliderConfig.scoreThreshold}
                onChange={(val) => handleSliderConfigChange("scoreThreshold", val)}
                onChangeEnd={applySliderConfig}
              >
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb />
              </Slider>
              <Text mt={2}>{sliderConfig.scoreThreshold}</Text>
            </FormControl>
            <FormControl>
              <FormLabel>Model Choice</FormLabel>
              <RadioGroup value={modelChoice} onChange={handleModelChoiceChange}>
                <Stack direction="row">
                  <Radio value="predefined">Predefined</Radio>
                  <Radio value="custom">Custom</Radio>
                </Stack>
              </RadioGroup>
            </FormControl>
            {modelChoice === "predefined" ? (
              <FormControl>
                <FormLabel>Model</FormLabel>
                <Select name="model" value={model} onChange={handleModelChange} placeholder="Select a model">
                  {models.map((modelName) => (
                    <option key={modelName} value={modelName}>
                      {modelName}
                    </option>
                  ))}
                </Select>
              </FormControl>
            ) : (
              <FormControl>
                <FormLabel>Upload Model</FormLabel>
                <Input type="file" ref={inputModel} accept=".onnx" onChange={(e) => loadModel(e.target.files[0])} />
              </FormControl>
            )}
          </VStack>
        </GridItem>
        <GridItem>
          <VStack spacing={4}>
            <Heading as="h1" size="xl">
              Card recognition using YOLOv8
            </Heading>
            <Text>
              Serving: <code className="code">{model}</code>
            </Text>

            <Box id="image-container" position="relative" display="inline-block" p={2}>
              <Image
                ref={imageRef}
                src={image}
                alt=""
                style={{
                  width: "100%",
                  maxWidth: "720px",
                  maxHeight: "500px",
                  borderRadius: "10px",
                }}
                display={image ? "block" : "none"}
                onLoad={handleImageLoad}
              />
              <canvas
                id="canvas"
                width={modelInputShape[2]}
                height={modelInputShape[3]}
                ref={canvasRef}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                }}
              />
            </Box>

            <input
              type="file"
              ref={inputImage}
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => {
                if (image) {
                  URL.revokeObjectURL(image);
                  setImage(null);
                  setIsImageLoaded(false);
                }

                const url = URL.createObjectURL(e.target.files[0]);
                imageRef.current.src = url;
                setImage(url);
              }}
            />
            <HStack spacing={4}>
              <Button
                onClick={() => {
                  inputImage.current.click();
                }}
              >
                Open local image
              </Button>
              {image && (
                <Button
                  onClick={() => {
                    inputImage.current.value = "";
                    imageRef.current.src = "#";
                    URL.revokeObjectURL(image);
                    setImage(null);
                    setIsImageLoaded(false);
                  }}
                >
                  Close image
                </Button>
              )}
            </HStack>
          </VStack>
        </GridItem>
      </Grid>
    </Box>
  );
};

export default App;
