import { LayerVariable } from "../variables";
export interface TransformerLayerArgs extends LayerArgs {
  numHeads: number;
  depth: number;
  pool: boolean;
  padSize: number;
}

class TransformerLayer extends tf.layers.Layer {
/** @nocollapse */
static className = "TransformerLayer";

private numHeads: number;
private depth: number;
private pool: boolean;
private padSize: number;

private inputDenseWeight: LayerVariable = null;
private inputDenseBias: LayerVariable = null;
private queryDenseWeight: LayerVariable = null;
private queryDenseBias: LayerVariable = null;
private keyDenseWeight: LayerVariable = null;
private keyDenseBias: LayerVariable = null;
private valueDenseWeight: LayerVariable = null;
private valueDenseBias: LayerVariable = null;
private denseWeight: LayerVariable = null;
private denseBias: LayerVariable = null;
private ffDense1Weight: LayerVariable = null;
private ffDense1Bias: LayerVariable = null;
private ffDense2Weight: LayerVariable = null;
private ffDense2Bias: LayerVariable = null;

readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier = "glorotNormal";
private weightsInitializer: Initializer;

constructor(args: TransformerLayerArgs) {
  super(args);

  this.numHeads = args.numHeads;
  this.depth = args.depth;
  this.pool = args.pool;
  this.padSize = args.padSize;

  if (this.depth % this.numHeads != 0) {
    throw new Error(
      `Assertion error : depth(${this.depth}) % numHead(${this.numHeads}) != 0 `
    );
  }

  this.inputDenseWeight = null;
  this.inputDenseBias = null;

  this.queryDenseWeight = null;
  this.queryDenseBias = null;

  this.keyDenseWeight = null;
  this.keyDenseBias = null;

  this.valueDenseWeight = null;
  this.valueDenseBias = null;

  this.denseWeight = null;
  this.denseBias = null;

  this.ffDense1Weight = null;
  this.ffDense1Bias = null;

  this.ffDense2Weight = null;
  this.ffDense2Bias = null;

  this.weightsInitializer = getInitializer(this.DEFAULT_KERNEL_INITIALIZER);
}

public build(inputShape: Shape[]) {
  const inputFeatSize = inputShape[0][inputShape[0].length - 1]!;

  this.inputDenseWeight = this.addWeight(
    "inputDenseWeight",
    [inputFeatSize, this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );
  this.inputDenseBias = this.addWeight(
    "inputDenseBias",
    [this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );

  this.queryDenseWeight = this.addWeight(
    "queryDenseWeight",
    [this.depth, this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );
  this.queryDenseBias = this.addWeight(
    "queryDenseBias",
    [this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );

  this.keyDenseWeight = this.addWeight(
    "keyDenseWeight",
    [this.depth, this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );
  this.keyDenseBias = this.addWeight(
    "keyDenseBias",
    [this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );

  this.valueDenseWeight = this.addWeight(
    "valueDenseWeight",
    [this.depth, this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );
  this.valueDenseBias = this.addWeight(
    "valueDenseBias",
    [this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );

  this.denseWeight = this.addWeight(
    "denseWeight",
    [this.depth, this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );
  this.denseBias = this.addWeight(
    "denseBias",
    [this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );

  this.ffDense1Weight = this.addWeight(
    "ffDense1Weight",
    [this.depth, this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );
  this.ffDense1Bias = this.addWeight(
    "ffDense1Bias",
    [this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );

  this.ffDense2Weight = this.addWeight(
    "ffDense2Weight",
    [this.depth, this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );
  this.ffDense2Bias = this.addWeight(
    "ffDense2Bias",
    [this.depth],
    "float32",
    this.weightsInitializer,
    undefined,
    true
  );

  this.built = true;
}

computeOutputShape(inputShape: tf.Shape[]): tf.Shape {
  if (this.pool) {
    return [inputShape[0][0], this.depth];
  } else {
    return [inputShape[0][0], inputShape[0][1], this.depth];
  }
}

call(
  inputs: Tensor | Tensor[],
  kwargs: { [key: string]: any }
): tf.Tensor | tf.Tensor[] {
  return tidy(() => {
    this.invokeCallHook(inputs, kwargs);
    const batchSize = inputs[0].shape[0];

    // Bring the input size to a lower size to have a smaller model
    // Also to have the [batch, token, depth] => [batch*token, depth] which can go in a dense layer
    const flatInput = inputs[0].reshape([this.padSize * batchSize, -1]); // [B, toks, emb] => [B*toks, emb]
    const flatScaledInput = K.dot(
      flatInput,
      this.inputDenseWeight!.read()
    ).add(this.inputDenseBias!.read()); // [B*toks, emb] => [B*toks, emb]
    const scaledInput = flatScaledInput.reshape([
      batchSize,
      this.padSize,
      -1,
    ]); // [B*toks, emb] => [B, toks, depth]

    // MultiHead Attention
    const flatQuery = K.dot(
      flatScaledInput,
      this.queryDenseWeight!.read()
    ).add(this.queryDenseBias!.read()); // [B*toks, emb] => [B*toks, emb]
    const flatKey = K.dot(flatScaledInput, this.keyDenseWeight!.read()).add(
      this.keyDenseBias!.read()
    ); // [B*toks, emb] => [B*toks, emb]
    const flatValue = K.dot(
      flatScaledInput,
      this.valueDenseWeight!.read()
    ).add(this.valueDenseBias!.read()); // [B*toks, emb] => [B*toks, emb]

    const query = flatQuery.reshape([batchSize, this.padSize, -1]); // [B*toks, emb] => [B, toks, depth]
    const key = flatKey.reshape([batchSize, this.padSize, -1]); // [B*toks, emb] => [B, toks, depth]
    const value = flatValue.reshape([batchSize, this.padSize, -1]); // [B*toks, emb] => [B, toks, depth]

    const queryT = transpose(
      query.reshape([
        batchSize,
        -1,
        this.numHeads,
        this.depth / this.numHeads,
      ]),
      [0, 2, 1, 3]
    ); // [B, toks, emb] => [B, nHeads, toks, depth//nHeads]
    const keyT = transpose(
      key.reshape([batchSize, -1, this.numHeads, this.depth / this.numHeads]),
      [0, 2, 1, 3]
    ); // [B, toks, emb] => [B, nHeads, toks, depth//nHeads]
    const valueT = transpose(
      value.reshape([
        batchSize,
        -1,
        this.numHeads,
        this.depth / this.numHeads,
      ]),
      [0, 2, 1, 3]
    ); // [B, toks, emb] => [B, nHeads, toks, depth//nHeads]

    // Scaled Dot product Attention
    // TODO Need to bring matMul from tfjs-core
    const matmul_qk = matMul(queryT, keyT, false, true); // [B, nHeads, toks, depth//nHeads] => [B, nHeads, toks, toks]

    let logits = matmul_qk.div(sqrt(cast(this.depth, "float32"))); // [B, nHeads, toks, toks] => [B, nHeads, toks, toks]

    const toBroadcastMask = inputs[1].expandDims(1).expandDims(1); // [B, toks] => [B, 1, 1, toks] : 1, 1 useful for broadcast then tfjs will broadcast to add to the [B, nHeads, toks, toks] logits
    logits = logits.add(scalar(1.0).sub(toBroadcastMask).mul(-1e9)); // [B, nHeads, toks, toks] => [B, nHeads, toks, toks] : with -inf where mask was one

    const attentionWeights = softmax(logits, -1); // [B, nHeads, toks, toks] => [B, nHeads, toks, toks]
    const scaledAttention = matMul(attentionWeights, valueT, true, false);

    // Reshape & Final attention
    const scaledAttentionT = transpose(scaledAttention, [0, 2, 1, 3]);
    const concatAttention = scaledAttentionT.reshape([
      scaledAttentionT.shape[0],
      -1,
      this.depth,
    ]);

    const flattenConcatAttention = concatAttention.reshape([
      batchSize * this.padSize,
      -1,
    ]);
    const flattenAttention = K.dot(
      flattenConcatAttention,
      this.denseWeight!.read()
    ).add(this.denseBias!.read());

    const attention = flattenAttention.reshape([batchSize, this.padSize, -1]);
    // Norm & Apply attention
    const normalizedLatent = tf.layers
      .layerNormalization()
      .apply(attention.add(scaledInput));

    // FeedForward
    const flattenNormalizedLatent = normalizedLatent.reshape([
      batchSize * this.padSize,
      -1,
    ]);
    const flatFf1 = K.dot(
      flattenNormalizedLatent,
      this.ffDense1Weight!.read()
    ).add(this.ffDense1Bias!.read());
    const flatRff1 = leakyReLU().apply(flatFf1);
    const flatFf2 = K.dot(flatRff1, this.ffDense2Weight!.read()).add(
      this.ffDense2Bias!.read()
    );
    const flatDff2 = dropout({ rate: 0.2 }).apply(flatFf2);

    const dff2 = flatDff2.reshape([batchSize, this.padSize, -1]);
    // Add&Norm
    const output = tf.layers
      .layerNormalization()
      .apply(normalizedLatent.add(dff2));

    if (this.pool) {
      return output.mean(1);
    } else {
      return output;
    }
  });
}

getConfig(): serialization.ConfigDict {
  const config = super.getConfig();
  Object.assign(config, {
    numHeads: this.numHeads,
    depth: this.depth,
  });
  return config;
}
}