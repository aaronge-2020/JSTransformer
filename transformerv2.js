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
  constructor(config) {
    super(config);
    this.dModel = config.dModel;
  }

    // Define serialization logic
    getConfig() {
      const config = super.getConfig();
      
      Object.assign(config, {
        dModel: this.dModel,
      });
      return config;
    }

  static fromConfig(cls, config) {
    return new cls(config);
  }

  static get className() {
    return "MultiplyLayer";
  }

  computeOutputShape(inputShape) {
    return inputShape; // The output shape is the same as the input shape
  }

  call(inputs, kwargs) {
    let input = inputs;
    if (Array.isArray(inputs)) {
      input = inputs[0];
    }

    return tf.mul(input, tf.sqrt(tf.scalar(this.dModel, "float32")));
  }

  getClassName() {
    return "MultiplyLayer";
  }
}

tf.serialization.registerClass(MultiplyLayer);

class AddLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  static fromConfig(cls, config) {
    return new cls(config);
  }

  static get className() {
    return "AddLayer";
  }

  computeOutputShape(inputShape) {
    return inputShape; // The output shape is the same as the input shape
  }

  call(inputs, kwargs) {
    let [input1, input2] = inputs;

    if (input1 instanceof tf.Tensor && input2 instanceof tf.Tensor) {
      return tf.add(input1, input2);
    } else {
      throw new Error('Both input1 and input2 should be instances of tf.Tensor.');
    }
  }

  getClassName() {
    return "AddLayer";
  }
}

tf.serialization.registerClass(AddLayer);

function applyEmbeddingAndPosEncoding(inputLanguage, input_vocab_size, dModel, dropout_rate = 0.1) {
  return tf.tidy(() => {
    const embedding_enc = tf.layers.embedding({
      inputDim: input_vocab_size,
      outputDim: dModel,
      maskZero: true,
    });

    const posEncoding_enc = positionalEncoding(input_vocab_size, dModel);
    const length = inputLanguage.shape[1];
  
    let x_pos_enc = embedding_enc.apply(inputLanguage);
  
    if (x_pos_enc instanceof tf.Tensor) {
      
      x_pos_enc = tf.mul(x_pos_enc, tf.sqrt(tf.scalar(dModel, "float32")));
  
      const posEncodingSliced = posEncoding_enc.slice([0, 0], [length, -1]);
  
      
      x_pos_enc = tf.add(posEncodingSliced, x_pos_enc);
  
      // Use tf.layers.dropout directly
      x_pos_enc = tf.layers.dropout({ rate: dropout_rate }).apply(x_pos_enc);
  
      return x_pos_enc;
    } else {
      return x_pos_enc;  // This could be a symbolic tensor
    }
  });
}


class TransformerModel {
  constructor(
    num_layers,
    dModel,
    numHeads,
    dff,
    input_vocab_size,
    target_vocab_size,
    dropout_rate,
    contextLen,
    targetLen,
    max_tokens = 60
  ) {
    this.num_layers = num_layers;
    this.dModel = dModel;
    this.numHeads = numHeads;
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
      this.dModel,
      this.dropout_rate
    );

    const enc_output = applyEncoderLayer(
      x_pos_enc,
      this.num_layers,
      this.dModel,
      this.numHeads,
      this.dff
    );

    let pos_embedding_dec = applyEmbeddingAndPosEncoding(
      outputLanguage,
      this.target_vocab_size,
      this.dModel,
      this.dropout_rate
    );

    const dec_output = applyDecoderLayer(
      enc_output,
      pos_embedding_dec,
      this.num_layers,
      this.dModel,
      this.numHeads,
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
  dModel,
  numHeads,
  dff,
  dropout_rate = 0.1
) {
  return tf.tidy(() => {
    const attentionLayers = Array.from({ length: num_layers }, () =>
      createBaseAttentionLayer(dModel, numHeads, dropout_rate, false)
    );
    const feedForwardLayers = Array.from({ length: num_layers }, () =>
      createFeedForwardLayer(dModel, dff, dropout_rate)
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
  dModel,
  numHeads,
  dff,
  dropout_rate = 0.1
) {
  if (dModel % numHeads !== 0) {
    throw new Error("dModel must be divisible by numHeads");
  }

  return tf.tidy(() => {
    const causalSelfAttentionLayers = Array.from({ length: num_layers }, () =>
      createCausalSelfAttentionLayer(dModel, numHeads, dropout_rate)
    );
    const crossAttentionLayers = Array.from({ length: num_layers }, () =>
      createCrossAttentionLayer(dModel, numHeads, dropout_rate)
    );
    const feedForwardLayers = Array.from({ length: num_layers }, () =>
      createFeedForwardLayer(dModel, dff, dropout_rate)
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

function createCausalSelfAttentionLayer(dModel, numHeads, dropout_rate) {
  const causalSelfAttentionLayer = createBaseAttentionLayer(
    dModel,
    numHeads,
    dropout_rate,
    true
  );
  return function (input) {
    return causalSelfAttentionLayer(input, input, input);
  };
}

function createCrossAttentionLayer(dModel, numHeads, dropout_rate) {
  const crossAttentionLayer = createBaseAttentionLayer(
    dModel,
    numHeads,
    dropout_rate,
    false
  );
  return function (input, context) {
    return crossAttentionLayer(input, context, context);
  };
}

function createFeedForwardLayer(dModel, dff, dropout_rate) {
  const seq = tf.sequential();
  seq.add(
    tf.layers.dense({ units: dff, activation: "relu", inputDim: dModel })
  );
  seq.add(tf.layers.dense({ units: dModel }));
  seq.add(tf.layers.dropout({ rate: dropout_rate }));
  const layerNorm = tf.layers.layerNormalization();

  return function (input) {
    return tf.tidy(() => { // Wrap the function body in tf.tidy
      const seqOutput = seq.apply(input);
      
      if (input instanceof tf.Tensor) {
        const output = tf.add(input, seqOutput);
        // Explicitly dispose of intermediate tensors if they are not needed
        seqOutput.dispose();
        
        return layerNorm.apply(output);
      } else {
        // Purely for when I'm running model.compile() to get the model summary
        return layerNorm.apply(seqOutput);
      }
    });
  };
}

function createBaseAttentionLayer(dModel, numHeads, dropout_rate, causal) {
  // Initialize necessary layers and variables here
  const mha = createMultiHeadAttentionLayer(dModel, numHeads, causal);
  const layernorm = tf.layers.layerNormalization();
  const add = tf.layers.add();

  return function (q, k, v) {
    return tf.tidy(() => {  // Wrap the function body in tf.tidy
      // Implement the operation of the layer here
      const attnOutput = mha(q, k, v);
      const addOutput = add.apply([q, attnOutput]);
      const output = layernorm.apply(addOutput);

      return output;
    });
  };
}


function createMultiHeadAttentionLayer(dModel, numHeads, causal) {
  return tf.tidy(() => { // Wrap the function body in tf.tidy
    if (dModel % numHeads !== 0) {
      throw new Error("dModel must be divisible by numHeads");
    }

    const depth = Math.floor(dModel / numHeads);

    // Reuse layers if they exist and have the same dimensions
    const wq = tf.layers.dense({ units: dModel });
    const wk = tf.layers.dense({ units: dModel });
    const wv = tf.layers.dense({ units: dModel });

    return function (q, k, v) {
      return tf.tidy(() => { // Another tf.tidy for this inner function
        const qProcessed = wq.apply(q);
        const kProcessed = wk.apply(k);
        const vProcessed = wv.apply(v);

        const splitHeadsAndComputeAttention = new SplitHeadsAndComputeAttention({
          dModel: dModel,
          numHeads: numHeads,
          depth: depth,
          causal: causal,
        });

        const output = splitHeadsAndComputeAttention.apply([qProcessed, kProcessed, vProcessed, q]);

        return output;
      });
    };
  });
}

// I need to have this class or else my model will not compile, because I will be doing matrix operations with symbolic tensors
class SplitHeadsAndComputeAttention extends tf.layers.Layer {
  constructor(config, kwargs) {
    super(config);
    this.dModel = config.dModel;
    this.numHeads = config.numHeads;
    this.depth = config.depth;
    this.causal = config.causal;
  }
  static fromConfig(cls, config) {
    return new cls(config);
  }

  // Define serialization logic
  getConfig() {
    const config = super.getConfig();
    
    Object.assign(config, {
      dModel: this.dModel,
      numHeads: this.numHeads,
      depth: this.depth,
      causal: this.causal,
    });
    return config;
  }

  static get className() {
    return "SplitHeadsAndComputeAttention";
  }

  getClassName() {
    return "SplitHeadsAndComputeAttention";
  }

  computeOutputShape(inputShape) {
    return inputShape[3];
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      let [q, k, v, x] = inputs;
      const batchSize = x.shape[0];

      [q, k, v] = [
        this.splitHeads(q, batchSize, this.numHeads, this.depth),
        this.splitHeads(k, batchSize, this.numHeads, this.depth),
        this.splitHeads(v, batchSize, this.numHeads, this.depth),
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
        this.dModel,
      ]);

      x = tf.add(x, concatAttention);

      return x;
    });
  }

  splitHeads(x, batch_size, numHeads, depth) {
    return tf.tidy(() => {
      const reshaped = tf.reshape(x, [batch_size, -1, numHeads, depth]);
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
  // Note: Assumes last axis is the 'class' axis, and index 0 is used for padding or special tokens
  
  return tf.tidy(() => { // Automatically clean up tensors
    // Find the index of the maximum value along axis 2 for both label and prediction
    const predArgMax = tf.argMax(pred, 2);
    const labelArgMax = tf.argMax(label, 2);
  
    // Check element-wise equality between the label and prediction
    let match = tf.equal(labelArgMax, predArgMax);
  
    // Create a mask where label is not equal to 0
    const mask = tf.notEqual(labelArgMax, 0);
  
    // Update the match tensor to only include positions where the mask is true
    match = tf.logicalAnd(match, mask);
  
    // Cast the boolean tensors to float32 for mathematical operations
    const castedMatch = tf.cast(match, "float32");
    const castedMask = tf.cast(mask, "float32");
  
    // Calculate the accuracy by summing up the matches and dividing by the sum of the mask
    const accuracy = tf.sum(castedMatch).div(tf.sum(castedMask));
    
    accuracy.print();

    // Return the accuracy tensor
    return accuracy;
  });
}


function maskedLoss(label, pred) {
  // Automatically dispose of intermediate tensors
  return tf.tidy(() => {
    // Create a mask where label is not equal to 0
    const mask = tf.notEqual(label, 0);

    // Compute the softmax cross-entropy loss
    const lossObject = tf.losses.softmaxCrossEntropy(label, pred, true, "none");

    // Cast the mask tensor to the same dtype as the lossObject for multiplication
    const castedMask = tf.cast(mask, lossObject.dtype);

    // Multiply the loss by the mask to ignore irrelevant tokens
    let loss = tf.mul(lossObject, castedMask);

    // Check if mask sum is greater than zero to avoid division by zero
    const maskSum = tf.sum(castedMask);
    if (maskSum.dataSync()[0] === 0) {
      throw new Error("Sum of mask is zero, division by zero would occur");
    }

    // Calculate the final loss by summing up the masked losses and dividing by the sum of the mask
    loss = tf.sum(loss).div(maskSum);

    // Return the final loss tensor
    return loss;
  });
}


function createMiniBatches(data, batchSize) {
  const miniBatches = [];
  for (let i = 0; i < data.length; i += batchSize) {
    const miniBatch = data.slice(i, i + batchSize);
    miniBatches.push(miniBatch);
  }
  return miniBatches;
}

export {
  // Transformer,
  TransformerModel,
  maskedLoss,
  maskedAccuracy,
  createMiniBatches,
};
