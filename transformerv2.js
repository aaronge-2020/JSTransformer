const tf = await import("https://esm.sh/@tensorflow/tfjs@4.10.0");

function positionalEncoding(length, depth) {
  /*
      Generates the positional encoding for a transformer model.
  
      Args:
          length: Sequence length
          depth: Dimensions of the embedding vector
  
  */

  depth = depth / 2;

  const positions = tf.range(0, length).reshape([-1, 1]); // (seq, 1)
  const depths = tf.range(0, depth).div(depth).reshape([1, -1]); // (1, depth)

  const angleRates = tf.pow(10000, depths.mul(-1)); // (1, depth)
  const angleRads = positions.mul(angleRates); // (pos, depth)

  const sinAngleRads = tf.sin(angleRads);
  const cosAngleRads = tf.cos(angleRads);

  const posEncoding = tf.concat([sinAngleRads, cosAngleRads], 1); // Concatenate along axis -1 (1 in JS)

  return posEncoding.toFloat();
}

class MultiplyLayer extends tf.layers.Layer {
  constructor(d_model) {
    super({});
    this.d_model = d_model;
  }

  computeOutputShape(inputShape) {
    return inputShape; // The output shape is the same as the input shape
  }

  call(inputs, kwargs) {
    let input = inputs;
    if (Array.isArray(inputs)) {
      input = inputs[0];
    }

    return tf.mul(input, tf.sqrt(tf.scalar(this.d_model, "float32")));
  }

  getClassName() {
    return "MultiplyLayer";
  }
}

tf.serialization.registerClass(MultiplyLayer);

class AddLayer extends tf.layers.Layer {
  constructor(addedValue) {
    super({});
    this.addedValue = addedValue;
  }

  computeOutputShape(inputShape) {
    return inputShape; // The output shape is the same as the input shape
  }

  call(inputs, kwargs) {
    let input = inputs;
    if (Array.isArray(inputs)) {
      input = inputs[0];
    }

    return tf.add(input, this.addedValue);
  }

  getClassName() {
    return "AddLayer";
  }
}
tf.serialization.registerClass(AddLayer);


function applyEmbeddingAndPosEncoding(
  inputLanguage,
  input_vocab_size,
  d_model,
  dropout_rate = 0.1
) {
  const embedding_enc = tf.layers.embedding({
    inputDim: input_vocab_size,
    outputDim: d_model,
    maskZero: true,
  });
  const posEncoding_enc = positionalEncoding(input_vocab_size, d_model);

  const length = inputLanguage.shape[1];

  let x_pos_enc = embedding_enc.apply(inputLanguage);

  const multiplyLayer = new MultiplyLayer(d_model);
  x_pos_enc = multiplyLayer.apply(x_pos_enc);

  const posEncodingSliced = posEncoding_enc.slice([0, 0], [length, -1]);

  const addLayer = new AddLayer(posEncodingSliced);
  x_pos_enc = addLayer.apply(x_pos_enc);

  const dropout_enc = tf.layers.dropout({ rate: dropout_rate });

  let x = dropout_enc.apply(x_pos_enc);

  return x;
}

class TransformerModel {
  constructor(
    num_layers,
    d_model,
    num_heads,
    dff,
    input_vocab_size,
    target_vocab_size,
    dropout_rate,
    contextLen,
    targetLen,
    max_tokens = 60
  ) {
    this.num_layers = num_layers;
    this.d_model = d_model;
    this.num_heads = num_heads;
    this.dff = dff;
    this.input_vocab_size = input_vocab_size;
    this.target_vocab_size = target_vocab_size;
    this.dropout_rate = dropout_rate;
    this.contextLen = contextLen;
    this.targetLen = targetLen;
    this.max_tokens = max_tokens;
    this.model = this.createModel();
  }

  createModel() {
    // Tensorflow will always add a dimension to your input to account for batchsize
    // The shape of the input is [sequence_length]
    // The actual input we will need to pass in is [batch_size, sequence_length]
    const inputLanguage = tf.input({ shape: [this.max_tokens] });
    const outputLanguage = tf.input({ shape: [this.max_tokens] });

    // Applying the embedding and positional encoding to the inputLanguage tensor.
    let x_pos_enc = applyEmbeddingAndPosEncoding(
      inputLanguage,
      this.input_vocab_size,
      this.d_model,
      this.dropout_rate
    );

    const enc_output = applyEncoderLayer(
      x_pos_enc,
      this.num_layers,
      this.d_model,
      this.num_heads,
      this.dff
    );

    let pos_embedding_dec = applyEmbeddingAndPosEncoding(
      outputLanguage,
      this.target_vocab_size,
      this.d_model,
      this.dropout_rate
    );

    const dec_output = applyDecoderLayer(
      enc_output,
      pos_embedding_dec,
      this.num_layers,
      this.d_model,
      this.num_heads,
      this.dff,
      this.target_vocab_size,
      this.dropout_rate
    );

    const final_layer = tf.layers.dense({
      inputDim: [null, null, null],
      units: this.target_vocab_size,
      computeOutputShape: [null, this.max_tokens, this.target_vocab_size],
      name: "final_layer",
    });

    // Final linear layer
    const logits = final_layer.apply(dec_output);

    return tf.model({
      inputs: [inputLanguage, outputLanguage],
      outputs: logits,
    });
  }
}

function applyEncoderLayer(
  x,
  num_layers,
  d_model,
  num_heads,
  dff,
  dropout_rate = 0.1
) {
  return tf.tidy(() => {
    const attentionLayers = Array.from({ length: num_layers }, () =>
      createBaseAttentionLayer(d_model, num_heads, dropout_rate, false)
    );
    const feedForwardLayers = Array.from({ length: num_layers }, () =>
      createFeedForwardLayer(d_model, dff, dropout_rate)
    );

    let output = x;
    for (let i = 0; i < num_layers; i++) {
      output = attentionLayers[i](output, output, output);
      output = feedForwardLayers[i](output);
    }

    return output;
  });
}

function applyDecoderLayer(
  enc_output,
  pos_embedding_dec,
  num_layers,
  d_model,
  num_heads,
  dff,
  dropout_rate = 0.1
) {
  if (d_model % num_heads !== 0) {
    throw new Error("d_model must be divisible by num_heads");
  }

  return tf.tidy(() => {
    const causalSelfAttentionLayers = Array.from({ length: num_layers }, () =>
      createCausalSelfAttentionLayer(d_model, num_heads, dropout_rate)
    );
    const crossAttentionLayers = Array.from({ length: num_layers }, () =>
      createCrossAttentionLayer(d_model, num_heads, dropout_rate)
    );
    const feedForwardLayers = Array.from({ length: num_layers }, () =>
      createFeedForwardLayer(d_model, dff, dropout_rate)
    );

    let output = pos_embedding_dec;
    for (let i = 0; i < num_layers; i++) {
      output = causalSelfAttentionLayers[i](output);
      output = crossAttentionLayers[i](output, enc_output);
      output = feedForwardLayers[i](output);
    }

    return output;
  });
}

function createCausalSelfAttentionLayer(d_model, num_heads, dropout_rate) {
  const causalSelfAttentionLayer = createBaseAttentionLayer(
    d_model,
    num_heads,
    dropout_rate,
    true
  );
  return function (input) {
    return causalSelfAttentionLayer(input, input, input);
  };
}

function createCrossAttentionLayer(d_model, num_heads, dropout_rate) {
  const crossAttentionLayer = createBaseAttentionLayer(
    d_model,
    num_heads,
    dropout_rate,
    false
  );
  return function (input, context) {
    return crossAttentionLayer(input, context, context);
  };
}

function createFeedForwardLayer(d_model, dff, dropout_rate) {
  const seq = tf.sequential();
  seq.add(
    tf.layers.dense({ units: dff, activation: "relu", inputDim: d_model })
  );
  seq.add(tf.layers.dense({ units: d_model }));
  seq.add(tf.layers.dropout({ rate: dropout_rate }));
  const layerNorm = tf.layers.layerNormalization();

  return function (input) {
    const seqOutput = seq.apply(input);
    if (input instanceof tf.Tensor) {
      const output = tf.add(input, seqOutput);
      return layerNorm.apply(output);
    } else {
      // Purely for when I'm running model.compile() to get the model summary
      return layerNorm.apply(seqOutput);
    }
  };
}

function createBaseAttentionLayer(d_model, num_heads, dropout_rate, causal) {
  // Initialize necessary layers and variables here
  const mha = createMultiHeadAttentionLayer(d_model, num_heads, causal);
  const layernorm = tf.layers.layerNormalization();
  const add = tf.layers.add();

  return function (q, k, v) {
    // Implement the operation of the layer here
    const attnOutput = mha(q, k, v);
    const addOutput = add.apply([q, attnOutput]);
    return layernorm.apply(addOutput);
  };
}

function createMultiHeadAttentionLayer(d_model, num_heads, causal) {
  if (d_model % num_heads !== 0) {
    throw new Error("d_model must be divisible by num_heads");
  }

  const depth = Math.floor(d_model / num_heads);

  const wq = tf.layers.dense({ units: d_model });
  const wk = tf.layers.dense({ units: d_model });
  const wv = tf.layers.dense({ units: d_model });

  return function (q, k, v) {
    const qProcessed = wq.apply(q);
    const kProcessed = wk.apply(k);
    const vProcessed = wv.apply(v);

    const splitHeadsAndComputeAttention = new SplitHeadsAndComputeAttention({
      d_model: d_model,
      num_heads: num_heads,
      depth: depth,
      causal: causal,
    });

    return splitHeadsAndComputeAttention.apply([
      qProcessed,
      kProcessed,
      vProcessed,
      q,
    ]);
  };
}

// I need to have this class or else my model will not compile, because I will be doing matrix operations with symbolic tensors
class SplitHeadsAndComputeAttention extends tf.layers.Layer {
  constructor({ d_model, num_heads, depth, causal = false }) {
    super({});
    this.d_model = d_model;
    this.num_heads = num_heads;
    this.depth = depth;
    this.causal = causal;
  }

  getClassName() {
    return "CustomAttentionLayer";
  }

  computeOutputShape(inputShape) {
    return inputShape[3];
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      let [q, k, v, x] = inputs;
      const batchSize = x.shape[0];

      [q, k, v] = [
        this.splitHeads(q, batchSize, this.num_heads, this.depth),
        this.splitHeads(k, batchSize, this.num_heads, this.depth),
        this.splitHeads(v, batchSize, this.num_heads, this.depth),
      ];

      let [scaledAttention, attentionWeights] = this.scaledDotProductAttention(
        q,
        k,
        v,
        this.causal,
        this.depth
      );

      const scaledAttentionTransposed = scaledAttention.transpose([0, 2, 1, 3]);
      const concatAttention = scaledAttentionTransposed.reshape([
        batchSize,
        -1,
        this.d_model,
      ]);

      x = tf.add(x, concatAttention);

      return x;
    });
  }

  splitHeads(x, batch_size, num_heads, depth) {
    return tf.tidy(() => {
      const reshaped = tf.reshape(x, [batch_size, -1, num_heads, depth]);
      return tf.transpose(reshaped, [0, 2, 1, 3]);
    });
  }

  scaledDotProductAttention(q, k, v, causal, depth) {
    const matmulQK = tf.matMul(q, k.transpose([0, 1, 3, 2]));
    let scaledAttentionLogits = matmulQK.div(tf.sqrt(depth));

    if (causal) {
      const seqLen =
        scaledAttentionLogits.shape[scaledAttentionLogits.shape.length - 2];
      const upperTriangular = tf.linalg.bandPart(
        tf.ones([seqLen, seqLen]),
        0,
        -1
      );
      const identityMatrix = tf.eye(seqLen);
      const causalMask = upperTriangular.sub(identityMatrix);
      scaledAttentionLogits = scaledAttentionLogits.add(
        causalMask.mul(tf.scalar(-1e9))
      );
    }

    const attentionWeights = tf.softmax(scaledAttentionLogits, -1);
    const output = tf.matMul(attentionWeights, v);

    return [output, attentionWeights];
  }
}
tf.serialization.registerClass(SplitHeadsAndComputeAttention);

function maskedAccuracy(label, pred) {
  // Log the label and prediction tensors to the console
  console.log(label, pred);
  
  // Find the index of the maximum value along axis 2 for both label and prediction
  let predArgMax = tf.argMax(pred, 2);
  let labelArgMax = tf.argMax(label, 2);
  
  // Check element-wise equality between the label and prediction
  let match = tf.equal(labelArgMax, predArgMax);

  // Create a mask where label is not equal to 0
  // This is useful for ignoring padding or other special tokens
  const mask = tf.notEqual(labelArgMax, 0);
  
  // Update the match tensor to only include positions where the mask is true
  match = tf.logicalAnd(match, mask);

  // Cast the boolean tensors to float32 for mathematical operations
  const castedMatch = tf.cast(match, 'float32');
  const castedMask = tf.cast(mask, 'float32');
  
  // Calculate the accuracy by summing up the matches and dividing by the sum of the mask
  const accuracy = tf.sum(castedMatch).div(tf.sum(castedMask));
  
  // Print the accuracy tensor
  accuracy.print();
  
  // Return the accuracy tensor
  return accuracy;
}

function maskedLoss(label, pred) {
  // Create a mask where label is not equal to 0
  // This is useful for ignoring padding or other special tokens
  const mask = tf.notEqual(label, 0);
  
  // Compute the softmax cross-entropy loss
  const lossObject = tf.losses.softmaxCrossEntropy(label, pred, true, 'none');
  
  // Cast the mask tensor to the same dtype as the lossObject for multiplication
  const castedMask = tf.cast(mask, lossObject.dtype);
  
  // Multiply the loss by the mask to ignore irrelevant tokens
  let loss = tf.mul(lossObject, castedMask);

  // Calculate the final loss by summing up the masked losses and dividing by the sum of the mask
  loss = tf.sum(loss).div(tf.sum(castedMask));
  
  // Return the final loss tensor
  return loss;
}

export {
  // Transformer,
  TransformerModel,
  maskedLoss,
  maskedAccuracy,
};
