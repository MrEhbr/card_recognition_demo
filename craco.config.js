const CopyPlugin = require("copy-webpack-plugin");
const path = require("path");
const fs = require("fs");

module.exports = {
  webpack: {
    plugins: {
      add: [
        new CopyPlugin({
          // Use copy plugin to copy *.wasm to output folder.
          patterns: [{ from: "node_modules/onnxruntime-web/dist/*.wasm", to: "static/js/[name][ext]" }],
        }),
        {
          apply: (compiler) => {
            compiler.hooks.beforeRun.tapAsync("ListModelsPlugin", (compilation, callback) => {
              const modelDir = path.resolve(__dirname, "public/model");
              const models = fs.readdirSync(modelDir).filter((file) => file.endsWith(".onnx"));

              console.log("Models found:", models);
              // Create a string representing the array of models in JavaScript format
              const modelsArray = JSON.stringify(models);

              // Path to the React component
              const reactComponentPath = path.resolve(__dirname, "src/App.js");

              // Read the React component
              let componentCode = fs.readFileSync(reactComponentPath, "utf-8");

              // Replace the hard-coded models array
              componentCode = componentCode.replace(/modelFiles = \[.*\];/, `modelFiles = ${modelsArray};`);

              // Write back the updated component code
              fs.writeFileSync(reactComponentPath, componentCode, "utf-8");

              callback();
            });
          },
        },
      ],
    },
    configure: (config) => {
      // set resolve.fallback for opencv.js
      config.resolve.fallback = {
        fs: false,
        path: false,
        crypto: false,
      };
      return config;
    },
  },
};
