import {
  // Transformer,
  TransformerModel,
} from "./transformerv2.js";

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

const response = await fetch('https://aaronge-2020.github.io/JSTransformer/data.json');


const jsonData = await response.json();

const processedData = processJson(jsonData, MAX_TOKENS)

const pt_train = processedData.trainData.map((item) => item.pt)

const en_train = processedData.trainData.map((item) => item.en)

// const pt_train = processedData.trainData.map((batch) => batch.map((item) => item.pt))

// const en_train = processedData.trainData.map((batch) => batch.map((item) => item.en))

const batch_size = 10;


const train_x = tf.tensor(pt_train.slice(0, batch_size))
const train_y = tf.tensor(en_train.slice(0, batch_size))

const seq_len = en_train[0].length; // assuming all items have the same length

let word_probs_label = new Array(batch_size).fill(null).map(() =>
  new Array(seq_len).fill(null).map(() =>
    new Array(target_vocab_size).fill(0)
  )
);

en_train.slice(0,batch_size).forEach((batch, batchIndex) => {
  batch.forEach((token, tokenIndex) => {
    if (token < target_vocab_size) {
      word_probs_label[batchIndex][tokenIndex][token] = 1;
    }
  });
});

console.log(word_probs_label);


// The model will output [batch_size, seq_len, vocab_size] so we have to reshape word_probs_label to match that. The word_probs_label should have the probability of each word in the vocab for each token in the sequence. 
// The third dimension of word_probs_label should be the vocab size with each value being the probability of that word being the next word in the sequence. The current value of the second dimension is the index of the word in the vocab so that will have a probability of 1 while the rest will be 0.




model.summary();
const result = model.predict([train_x, train_y]).print();

await model.fit([train_x, train_y], tf.tensor(word_probs_label), {
  batch_size: batch_size,
})




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
