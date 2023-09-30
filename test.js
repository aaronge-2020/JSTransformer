import {
  // Transformer,
  TransformerModel,
  maskedAccuracy,
  maskedLoss,
  createMiniBatches,
} from "./transformerv2.js";

import { processJson, shiftTokens, wordsToIntTokens } from "./tokenizer.js";

function countOccurrences(arrOfArrs, target) {
  let count = 0;

  // Loop through each sub-array
  for (let i = 0; i < arrOfArrs.length; i++) {
    const subArray = arrOfArrs[i];

    // Loop through each element in the sub-array
    for (let j = 0; j < subArray.length; j++) {
      if (subArray[j] === target) {
        count++;
      }
    }
  }

  return count;
}

const tf = await import("https://esm.sh/@tensorflow/tfjs@4.10.0");

// Test the Transformer
const num_layers = 4;
const d_model = 256;

// The dff is the node size of the feed forward network
const dff = 128;
const num_heads = 8;
const dropout_rate = 0.1;

const input_vocab_size = 3000;
const target_vocab_size = 5500;

const MAX_TOKENS = 13;

// Sampling from the dataset
const start_point = 0;
const batch_size = 128;
const numb_of_batches = 125;
const numb_samples = batch_size * numb_of_batches;
const numb_epochs = 15;

const response = await fetch("http://127.0.0.1:5500/curated_dataset.json");

const jsonData = await response.json();

// Save this processed data to local storage to avoid reprocessing and save time


let processedData = processJson(
  jsonData.slice(start_point, start_point + numb_samples),
  "English",
  "Spanish",
  input_vocab_size,
  target_vocab_size,
  MAX_TOKENS
);

const tokenizers = processedData.tokenizers;

const en_train = processedData.trainData.map((item) => item.English);

// const en_validation = processedData.validationData.map((item) => item.English)

const sp_train = processedData.trainData.map((item) => item.Spanish);

// const sp_validation = processedData.validationData.map((item) => item.Spanish)

processedData = undefined; 

const training_dataset = shiftTokens(sp_train, 2);

const target_lang_input_train = training_dataset[0];

const target_lang_label_train = training_dataset[1];

// const validation_dataset = shiftTokens(sp_validation, 2)

// const target_lang_input_validation = validation_dataset[0]

// const target_lang_label_validation = validation_dataset[1]

const seq_len = target_lang_label_train[0].length; // assuming all items have the same length

let word_probs_label_train = new Array(sp_train.length)
  .fill(null)
  .map(() =>
    new Array(seq_len)
      .fill(null)
      .map(() => new Array(target_vocab_size).fill(0))
  );

target_lang_label_train.forEach((batch, batchIndex) => {
  batch.forEach((token, tokenIndex) => {
    if (token < target_vocab_size) {
      word_probs_label_train[batchIndex][tokenIndex][token] = 1;
    }
  });
});

// let word_probs_label_validation = new Array(sp_validation.length).fill(null).map(() =>
//   new Array(seq_len).fill(null).map(() =>
//     new Array(target_vocab_size).fill(0)
//   )
// );

// target_lang_label_validation.forEach((batch, batchIndex) => {
//   batch.forEach((token, tokenIndex) => {
//     if (token < target_vocab_size) {
//       word_probs_label_validation[batchIndex][tokenIndex][token] = 1;
//     }
//   });
// });

// const transformerModel = new TransformerModel(
//   num_layers,
//   d_model,
//   num_heads,
//   dff,
//   input_vocab_size,
//   target_vocab_size,
//   dropout_rate,
//   input_vocab_size,
//   target_vocab_size,
//   MAX_TOKENS
// );
// const model = transformerModel.model;
// model.compile({
//   loss: maskedLoss,
//   optimizer: "adam",
//   metrics: [maskedAccuracy],
// });


// Create mini-batches for training and validation data
const en_train_batches = createMiniBatches(en_train, batch_size);
const target_lang_input_train_batches = createMiniBatches(
  target_lang_input_train,
  batch_size
);
const word_probs_label_train_batches = createMiniBatches(
  word_probs_label_train,
  batch_size
);



const loadedModel = await tf.loadLayersModel(
  "http://127.0.0.1:5500/my-model-epoch-2-batch-41.json"
);

loadedModel.compile({
  loss: maskedLoss,
  optimizer: "adam",
  metrics: maskedAccuracy,
// });

// // Train the model

// {
//   for (let j = 0; j < numb_epochs; j++) {
//     for (let i = 1; i < en_train_batches.length - 1; i++) {
      
//       const train_x_batch = tf.tensor(en_train_batches[i]);
//       const train_y_batch = tf.tensor(target_lang_input_train_batches[i]);
//       const labels_batch = tf.tensor(word_probs_label_train_batches[i]);

//       try {
//         // Train the model on the current batch
//         await loadedModel.trainOnBatch(
//           [train_x_batch, train_y_batch],
//           labels_batch
//         );

//         // save the model every 20 batches
//         if (i % 20 === 0){
//           await loadedModel.save('downloads://my-model' + '-epoch-' + (j + 1) + '-batch-' + (i));
//         }


//       } catch (error) {
//         console.error(error);
//       }

//       // Dispose tensors to free memory
//       train_x_batch.dispose();
//       train_y_batch.dispose();
//       labels_batch.dispose();

//       console.log(
//         `Batch ${i + 1} completed. ${
//           en_train_batches.length - i - 1
//         } batches remaining.`
//       );
//     }

//     console.log(
//       `Epoch ${j + 1} completed. ${numb_epochs - j - 1} epochs remaining.`
//     );
//   }
// }
// // await model.save('downloads://my-model');
// // loadedModel.predict([tf.tensor(en_train_batches[0]), tf.tensor(target_lang_input_train_batches[0])]).print();
