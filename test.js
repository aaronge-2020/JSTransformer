import {
  Transformer
} from "./transformer.js";

import {
  processJson
} from "./tokenizer.js";


// Test the Transformer
const num_layers = 4
const d_model = 128
const dff = 512
const num_heads = 8
const dropout_rate = 0.1

const transformer = new Transformer(
  num_layers,
  d_model,
  num_heads,
  dff,
  1000,
  1000,
  0.1)

  const response = await fetch('./data.json');
  const jsonData = await response.json();

console.log(processJson(jsonData));

const tf = await import("https://esm.sh/@tensorflow/tfjs@4.10.0");
