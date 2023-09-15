import {
  Transformer,
  TransformerModel,
  MultiHeadAttention,
  Encoder
} from "./transformer.js";

import {
  processJson
} from "./tokenizer.js";

const tf = await import("https://esm.sh/@tensorflow/tfjs@4.10.0");

// Test the Transformer
const num_layers = 4
const d_model = 64

// The dff is the node size of the feed forward network
const dff = 128
const num_heads = 8
const dropout_rate = 0.1

const input_vocab_size = 45058
const target_vocab_size = 28021

const MAX_TOKENS = 60;

const transformerModel = new TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, input_vocab_size, target_vocab_size, MAX_TOKENS);
const model = transformerModel.model;
model.compile({
  loss: 'categoricalCrossentropy',
  optimizer: 'adam',
  metrics: ['accuracy']
});

const response = await fetch('https://aaronge-2020.github.io/data.json');


const jsonData = await response.json();

const processedData = processJson(jsonData, MAX_TOKENS)


function convertToTensor(arr) {
  arr = tf.tensor(arr); // Convert to 0-padded dense Tensor
  return arr;
;
}

const pt_train = processedData.trainData.map((item) => item.pt)

const en_train = processedData.trainData.map((item) => item.en)

// const pt_train = processedData.trainData.map((batch) => batch.map((item) => item.pt))

// const en_train = processedData.trainData.map((batch) => batch.map((item) => item.en))



const train_x = tf.expandDims(tf.tensor(pt_train[0]), 0)
const train_y = tf.expandDims(tf.tensor(en_train[0]), 0)

// model.fit([train_x, train_y], [train_x, train_y])

model.predict([train_x, train_y]).print();

console.log("hello");



// // Create an instance of the Encoder
// const encoder = new Encoder(2, 512, 8, 2048, 10000, 0.1);

// // Create some mock data (batch_size: 3, seq_len: 4)
// const mockData = tf.tensor([
//   [1, 2, 3, 4],
//   [5, 6, 7, 8],
//   [9, 10, 11, 12],
// ]);

// // Test the Encoder
// const output = encoder.apply(mockData);

// // Print the output shape (should be [3, 4, 512] if everything is set up correctly)
// output.print();


// // Define input, which has a size of 5 (not including batch dimension).
// const input = tf.input({shape: [5]});
// const input2 = tf.input({shape: [5]});


// // First dense layer uses relu activation.
// const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
// // Second dense layer uses softmax activation.
// const denseLayer2 = tf.layers.dense({units: 4, activation: 'softmax'});

// // Obtain the output symbolic tensor by applying the layers on the input.
// const output = denseLayer2.apply(denseLayer1.apply(input));

// // Create the model based on the inputs.
// const model = tf.model({inputs: [input, input2], outputs: output});

// // The model can be used for training, evaluation and prediction.
// // For example, the following line runs prediction with the model on
// // some fake data.
// model.predict([tf.ones([2, 5]),  tf.ones([2, 5])] ).print();
