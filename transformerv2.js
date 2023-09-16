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

class PositionalEmbedding extends tf.layers.Layer {
    constructor(vocabSize, dModel) {
        super({ trainable: true });
        this.dModel = dModel;
        this.embedding = tf.layers.embedding({
            inputDim: vocabSize,
            outputDim: dModel,
            maskZero: true,
        });
        this.posEncoding = positionalEncoding(vocabSize, dModel);
    }

    call(inputs) {
        let input = inputs;
        if (Array.isArray(input)) {
            input = input[0];
        }

        const length = input.shape[1];
        let x = this.embedding.apply(input);

        // This factor sets the relative scale of the embedding and positionalEncoding
        x = x.mul(tf.sqrt(tf.scalar(this.dModel, "float32")));

        // We need to slice the positional encoding, because the original shape is [2048, 512], where 2048 is the max sequence length and 512 is the embedding size
        const posEncodingSliced = this.posEncoding.slice([0, 0], [length, -1]);
        x = x.add(posEncodingSliced);

        return x;
    }

    getClassName() {
        return "PositionalEmbedding";
    }

}
class MultiHeadAttention extends tf.layers.Layer {
    constructor(d_model, num_heads, causal = false) {
        super({ trainable: true });
        this.num_heads = num_heads;
        this.d_model = d_model;
        this.causal = causal; // Add this line to include a causal flag
        if (d_model % this.num_heads !== 0) {
            throw new Error("d_model must be divisible by num_heads");
        }
        this.depth = Math.floor(d_model / this.num_heads);

        this.wq = tf.layers.dense({ units: d_model });
        this.wk = tf.layers.dense({ units: d_model });
        this.wv = tf.layers.dense({ units: d_model });
        this.dense = tf.layers.dense({ units: d_model });
    }

    scaledDotProductAttention(q, k, v) {
        const matmulQK = tf.matMul(q, k.transpose([0, 1, 3, 2]));
        const dk = k.shape[k.shape.length - 1];
        let scaledAttentionLogits = matmulQK.div(tf.sqrt(dk));

        // Apply causal mask if required
        if (this.causal) {
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

    splitHeads(x, batch_size) {
        const reshaped = x.reshape([batch_size, -1, this.num_heads, this.depth]);
        return reshaped.transpose([0, 2, 1, 3]);
    }

    call(inputs) {
        const [q, k, v] = inputs;
        const batchSize = q.shape[0];
        const qProcessed = this.wq.apply(q);
        const kProcessed = this.wk.apply(k);
        const vProcessed = this.wv.apply(v);

        const qSplit = this.splitHeads(qProcessed, batchSize);
        const kSplit = this.splitHeads(kProcessed, batchSize);
        const vSplit = this.splitHeads(vProcessed, batchSize);

        const [scaledAttention, attentionWeights] = this.scaledDotProductAttention(
            qSplit,
            kSplit,
            vSplit
        );

        const scaledAttentionTransposed = scaledAttention.transpose([0, 2, 1, 3]);
        const concatAttention = scaledAttentionTransposed.reshape([
            batchSize,
            -1,
            this.d_model,
        ]);
        const output = this.dense.apply(concatAttention);

        return [output, attentionWeights];
    }

    getClassName() {
        return "MultiHeadAttention";
    }

}

class BaseAttention extends tf.layers.Layer {
    constructor(d_model, num_heads, config, causal = false) {
        super({ trainable: true });

        this.mha = new MultiHeadAttention(d_model, num_heads, config, causal);
        this.layernorm = tf.layers.layerNormalization();
        this.add = tf.layers.add();
    }

    getClassName() {
        return "BaseAttention";
    }
}

class CrossAttention extends BaseAttention {
    constructor(d_model, num_heads, config) {
        super(d_model, num_heads, config);
    }
    call(inputs, kwargs) {
        const [x, context] = inputs;

        // Forward pass through MultiHeadAttention
        const [attnOutput, attnScores] = this.mha.apply(
            [x, context, context],
            null
        );

        // Cache the attention scores for later use or plotting
        this.lastAttnScores = attnScores;

        // Add the output to the original input (Residual connection)
        const addedOutput = this.add.apply([x, attnOutput]);

        // Layer normalization
        const normalizedOutput = this.layernorm.apply(addedOutput);

        return normalizedOutput;
    }


    getClassName() {
        return "CrossAttention";
    }
}

class GlobalSelfAttention extends BaseAttention {
    constructor(d_model, num_heads, config) {
        super(d_model, num_heads, config);
    }

    call(inputs, kwargs) {
        const x = inputs;

        // Forward pass through MultiHeadAttention
        const [attnOutput, attnScores] = this.mha.apply([x, x, x]);

        // Add the output to the original input (Residual connection)
        const addedOutput = this.add.apply([x, attnOutput]);

        // Layer normalization
        const normalizedOutput = this.layernorm.apply(addedOutput);

        return normalizedOutput;
    }

    getClassName() {
        return "GlobalSelfAttention";
    }
}

class CausalSelfAttention extends BaseAttention {
    constructor(d_model, num_heads, config) {
        super(d_model, num_heads, config, true); // Enable causal attention by setting the last argument to true
    }

    call(x) {
        // Multi-head attention layer with causal mask enabled
        const [attnOutput, _] = this.mha.apply([x, x, x]);

        // Add & normalize layer
        const addOutput = this.add.apply([x, attnOutput]);
        const output = this.layernorm.apply(addOutput);

        return output;
    }

    getClassName() {
        return "CausalSelfAttention";
    }
}
class FeedForward extends tf.layers.Layer {
    /*
  
      Implements the feedforward network used in the transformer.
  
      Args:
          dModel: Depth of the input vector
          dff: Hidden layer size
          dropoutRate: Dropout rate
  
      */

    constructor(dModel, dff, dropoutRate = 0.1) {
        super({ trainable: true });
        this.seq = tf.sequential();
        this.seq.add(
            tf.layers.dense({ units: dff, activation: "relu", inputDim: dModel })
        );
        this.seq.add(tf.layers.dense({ units: dModel }));
        this.seq.add(tf.layers.dropout({ rate: dropoutRate }));
        this.layerNorm = tf.layers.layerNormalization();
    }

    call(x) {
        const seqOutput = this.seq.apply(x);
        x = tf.add(x, seqOutput);
        x = this.layerNorm.apply(x);
        return x;
    }

    getClassName() {
        return "FeedForward";
    }
}

// The Encoder class extends the tf.layers.Layer class, allowing us to create a custom layer in TensorFlow.js.
class Encoder extends tf.layers.Layer {
    constructor(
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        dropout_rate = 0.1
    ) {
        super(); // Calling the super class constructor (tf.layers.Layer)

        // Storing the input parameters as class properties.
        this.d_model = d_model;
        this.num_layers = num_layers;

        // Initializing PositionalEmbedding layer with vocab_size and d_model.
        this.pos_embedding = new PositionalEmbedding(vocab_size, d_model);

        // Creating an array of EncoderLayer instances, with each instance being initialized with the provided parameters.
        this.enc_layers = Array.from(
            { length: num_layers },
            () => new EncoderLayer(d_model, num_heads, dff)
        );

        // Initializing a dropout layer with the specified dropout rate.
        this.dropout = tf.layers.dropout({ rate: dropout_rate });
    }

    // The call method is responsible for forward propagation in the Encoder layer.
    call(x) {
        // Applying the positional embedding to the input x.
        x = this.pos_embedding.apply(x); // Expected input shape: (batch_size, sequence_length), output shape: (batch_size, sequence_length, d_model)

        // Applying dropout to the output of the positional embedding layer.
        x = this.dropout.apply(x); // Applies dropout to reduce overfitting.

        // Iterating through the EncoderLayer instances and applying them sequentially to the input.
        for (let i = 0; i < this.num_layers; i++) {
            x = this.enc_layers[i].apply(x); // Expected input/output shape: (batch_size, sequence_length, d_model)
        }

        // Returning the output tensor.
        return x; // Output shape: (batch_size, sequence_length, d_model)
    }

    // Method to return the class name as a string.
    getClassName() {
        return "Encoder";
    }

    computeOutputShape() {
        return [null, this.max_tokens, this.d_model];
    }
}

// The EncoderLayer class, representing an individual layer within the encoder.
class EncoderLayer extends tf.layers.Layer {
    constructor(d_model, num_heads, dff, vocab_size) {
        super(); // Calling the super class constructor (tf.layers.Layer)

        // Initializing the global self attention and feed forward layers with the specified parameters.
        this.self_attention = new GlobalSelfAttention(d_model, num_heads);
        this.ffn = new FeedForward(d_model, dff);
    }

    // The call method is responsible for forward propagation in the EncoderLayer.
    call(x) {
        // Applying self attention to the input tensor x.
        x = this.self_attention.apply(x[0]); // Expected input/output shape: (batch_size, sequence_length, d_model)

        // Applying the feedforward neural network to the output of the self attention layer.
        x = this.ffn.apply(x); // Expected input/output shape: (batch_size, sequence_length, d_model)

        // Returning the output tensor.
        return x; // Output shape: (batch_size, sequence_length, d_model)
    }

    // Method to return the class name as a string.
    getClassName() {
        return "EncoderLayer";
    }
}

class DecoderLayer extends tf.layers.Layer {
    constructor(d_model, num_heads, dff, dropout_rate = 0.1, vocab_size) {
        super({ trainable: true });
        this.causalSelfAttention = new CausalSelfAttention(
            d_model,
            num_heads,
            dropout_rate
        );

        this.crossAttention = new CrossAttention(d_model, num_heads, dropout_rate);

        this.ffn = new FeedForward(d_model, dff);
        this.d_model = d_model;
        this.vocab_size = vocab_size;
    }

    call(inputs) {
        const [x, context] = inputs;
        let out = this.causalSelfAttention.apply(x);
        out = this.crossAttention.apply([out, context]);

        // Cache the last attention scores for plotting later
        this.lastAttnScores = this.crossAttention.lastAttnScores;

        out = this.ffn.apply(out);
        return out;
    }

    getClassName() {
        return "DecoderLayer";
    }

    computeOutputShape() {
        return [null, this.vocabSize, this.d_model];
    }

}

class Decoder extends tf.layers.Layer {
    constructor(
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        dropout_rate = 0.1
    ) {
        super({ trainable: true });
        this.d_model = d_model;
        this.num_layers = num_layers;

        this.pos_embedding = new PositionalEmbedding(vocab_size, d_model);
        this.dropout = tf.layers.dropout({ rate: dropout_rate });
        this.dec_layers = Array.from(
            { length: num_layers },
            () => new DecoderLayer(d_model, num_heads, dff, dropout_rate, vocab_size)
        );

        this.last_attn_scores = null;
    }

    call(inputs) {
        const [x, context] = inputs;

        let out = this.pos_embedding.apply(x);

        out = this.dropout.apply(out);

        for (let i = 0; i < this.num_layers; i++) {
            out = this.dec_layers[i].apply([out, context]);
        }

        this.last_attn_scores = this.dec_layers[this.num_layers - 1].lastAttnScores;

        return out;
    }

    getClassName() {
        return "Decoder";
    }

    computeOutputShape() {
        return [null, this.max_tokens, this.d_model];
    }

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
        // The shape of the input is [number_of_samples, sequence_length]
        // The actual input we will need to pass in is [batch_size, number_of_samples, sequence_length]
        const inputLanguage = tf.input({ shape: [this.max_tokens] });
        const outputLanguage = tf.input({ shape: [this.max_tokens] });


        // Initializing PositionalEmbedding layer with vocab_size and d_model.
        // const pos_embedding = new PositionalEmbedding(this.input_vocab_size, this.d_model);
        const embedding_enc = tf.layers.embedding({
            inputDim: this.input_vocab_size,
            outputDim: this.d_model,
            maskZero: true,
        });
        const posEncoding_enc = positionalEncoding(this.input_vocab_size, this.dModel);

        const length = inputLanguage.shape[1];

        let x_pos_enc = embedding_enc.apply(inputLanguage);

        // Assuming x_pos_enc is a tf.Tensor

        // Check if the input is a tf.Tensor
        if (x_pos_enc instanceof tf.Tensor) {
            x_pos_enc = tf.mul(x_pos_enc, tf.sqrt(tf.scalar(this.d_model, 'float32')));

            // For slicing and adding positional encoding
            const posEncodingSliced = posEncoding_enc.slice([0, 0], [length, -1]);
            x_pos_enc = tf.add(x_pos_enc, posEncodingSliced);
        }

        // Creating an array of EncoderLayer instances, with each instance being initialized with the provided parameters.
        const enc_layers = Array.from(
            { length: this.num_layers },
            () => new EncoderLayer(this.d_model, this.num_heads, this.dff)
        );

        // Initializing a dropout layer with the specified dropout rate.
        const dropout_enc = tf.layers.dropout({ rate: this.dropout_rate });


        // Applying dropout to the output of the positional embedding layer.
        let x = dropout_enc.apply(x_pos_enc); // Applies dropout to reduce overfitting.

        // Iterating through the EncoderLayer instances and applying them sequentially to the input.
        for (let i = 0; i < this.num_layers; i++) {
            x = enc_layers[i].apply(x); // Expected input/output shape: (batch_size, sequence_length, d_model)
        }

        const enc_output = x;


        const pos_embedding_dec = new PositionalEmbedding(this.target_vocab_size, this.d_model);
        const dropout_dec = tf.layers.dropout({ rate: this.dropout_rate });
        const dec_layers = Array.from(
            { length: this.num_layers },
            () => new DecoderLayer(this.d_model, this.num_heads, this.dff, this.dropout_rate, this.target_vocab_size)
        );

        let out = pos_embedding_dec.apply(outputLanguage);

        out = dropout_dec.apply(out);

        for (let i = 0; i < this.num_layers; i++) {
            out = dec_layers[i].apply([outputLanguage, enc_output]);
        }

        const dec_output = out;

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

export {
    positionalEncoding,
    PositionalEmbedding,
    MultiHeadAttention,
    CrossAttention,
    GlobalSelfAttention,
    CausalSelfAttention,
    Encoder,
    DecoderLayer,
    Decoder,
    // Transformer,
    TransformerModel,
};
