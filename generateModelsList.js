const fs = require("fs");
const path = require("path");

const modelDir = path.join(__dirname, "public", "model");
const outputFile = path.join(__dirname, "public", "modelList.json");

// Read the list of .onnx model files from the local directory
const models = fs.readdirSync(modelDir).filter((file) => file.endsWith(".onnx") && !file.startsWith("nms"));

// Write the model list to a JSON file
fs.writeFileSync(outputFile, JSON.stringify(models, null, 2), "utf-8");

console.log(`Model list generated at ${outputFile}`);
