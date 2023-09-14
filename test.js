import {
  Transformer,
  TransformerModel
} from "./transformer.js";

import {
  processJson
} from "./tokenizer.js";

const tf = await import("https://esm.sh/@tensorflow/tfjs@4.10.0");

// Test the Transformer
const num_layers = 4
const d_model = 128
const dff = 512
const num_heads = 8
const dropout_rate = 0.1

const input_vocab_size = 2483
const target_vocab_size = 2483

const transformerModel = new TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, input_vocab_size, target_vocab_size);
const model = transformerModel.model;
model.compile({
  loss: 'categoricalCrossentropy',
  optimizer: 'adam',
  metrics: ['accuracy']
});

const response = await fetch('http://127.0.0.1:5500/data.json');


const jsonData = await response.json();

const processedData = processJson(jsonData)
const MAX_TOKENS = 128;


function convertToTensor(arr) {
  arr = tf.tensor(arr); // Convert to 0-padded dense Tensor
  return arr;
;
}
function convertTo3DTensor(arrayOfArraysOfTensors) {
  // Convert the array of arrays of tensors to a 3D tensor
  const stackedTensors = arrayOfArraysOfTensors.map(arr => tf.stack(arr));
  return tf.stack(stackedTensors);
}

const pt_train = processedData.trainData.map((item) => convertToTensor(item.pt))

const en_train = processedData.trainData.map((item) => convertToTensor(item.en))

const train_x = tf.stack(pt_train)
const train_y = tf.stack(en_train)

model.fit([train_x, train_y])

console.log("hello");