import {
  // Transformer,
  TransformerModel,
  maskedAccuracy,
  maskedLoss
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

const input_vocab_size = 2000
const target_vocab_size = 2350

const MAX_TOKENS = 7;


// const transformerModel = new TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate, input_vocab_size, target_vocab_size, MAX_TOKENS);
// const model = transformerModel.model;
// model.compile({
//   loss: maskedLoss,
//   optimizer: 'adam',
//   metrics: [maskedAccuracy]
// });

const start_point = 1000;
const numb_samples = 3000;

const response = await fetch('http://127.0.0.1:5500/translation_pairs.json');



const jsonData = await response.json();

const processedData = processJson(jsonData.slice(start_point, start_point+numb_samples), "English", "Spanish", input_vocab_size, target_vocab_size, MAX_TOKENS)

const en_train = processedData.trainData.map((item) => item.English)

const sp_train = processedData.trainData.map((item) => item.Spanish)

// const pt_train = processedData.trainData.map((batch) => batch.map((item) => item.pt))

// const en_train = processedData.trainData.map((batch) => batch.map((item) => item.en))



const train_x = tf.tensor(en_train.slice(0, 10))
const train_y = tf.tensor(sp_train.slice(0, 10))

// const seq_len = output_language[0].length; // assuming all items have the same length

// let word_probs_label = new Array(batch_size).fill(null).map(() =>
//   new Array(seq_len).fill(null).map(() =>
//     new Array(target_vocab_size).fill(0)
//   )
// );

// output_language.slice(0,batch_size).forEach((batch, batchIndex) => {
//   batch.forEach((token, tokenIndex) => {
//     if (token < target_vocab_size) {
//       word_probs_label[batchIndex][tokenIndex][token] = 1;
//     }
//   });
// });

// console.log(word_probs_label);


// // The model will output [batch_size, seq_len, vocab_size] so we have to reshape word_probs_label to match that. The word_probs_label should have the probability of each word in the vocab for each token in the sequence. 
// // The third dimension of word_probs_label should be the vocab size with each value being the probability of that word being the next word in the sequence. The current value of the second dimension is the index of the word in the vocab so that will have a probability of 1 while the rest will be 0.


// function translateProbabilityToSentence(probability, detokenizer){
//   return probability.map((sentence) => sentence.map ( (word) => detokenizer[word.indexOf(Math.max(...word))]))
// }

// model.summary();
// const result = model.predict([train_x, train_y]).print();

// await model.fit([train_x, train_y], tf.tensor(word_probs_label), {
//   batch_size: batch_size,
// })


const loadedModel = await tf.loadLayersModel("http://127.0.0.1:5500/final-model.json")

loadedModel.predict([train_x, train_y]).print()