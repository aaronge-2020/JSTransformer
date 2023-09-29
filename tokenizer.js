const winkNLP = (await import("https://cdn.skypack.dev/wink-nlp")).default;
const mod = (await import("https://cdn.skypack.dev/wink-eng-lite-web-model"))
  .default;
const nlp = winkNLP(mod);

function buildVocabulary(
  tokenizedData,
  languageOne,
  languageTwo,
  maxVocabEn,
  maxVocabPt
) {
  let vocabularyEn = {};
  let vocabularyPt = {};

  tokenizedData.forEach((item) => {
    item[languageOne].forEach((token) => {
      vocabularyEn[token] = (vocabularyEn[token] || 0) + 1;
    });
    item[languageTwo].forEach((token) => {
      vocabularyPt[token] = (vocabularyPt[token] || 0) + 1;
    });
  });

  // Sort by frequency and take top maxVocab words
  vocabularyEn = Object.entries(vocabularyEn)
    .sort(([, a], [, b]) => b - a)
    .slice(0, maxVocabEn)
    .reduce((acc, [k, v]) => ({ ...acc, [k]: v }), {});
  vocabularyPt = Object.entries(vocabularyPt)
    .sort(([, a], [, b]) => b - a)
    .slice(0, maxVocabPt)
    .reduce((acc, [k, v]) => ({ ...acc, [k]: v }), {});

  return { vocabularyEn, vocabularyPt };
}

function convertTokensToIntegers(
  tokenizedData,
  vocabEn,
  vocabPt,
  languageOne,
  languageTwo,
  max_tokens = 50
) {
  const tokenToIntEn = { "<UNKNOWN>": 3 };
  const tokenToIntPt = { "<UNKNOWN>": 3 };
  const intToTokenLangOne = {
    0: "<PAD>",
    1: "<START>",
    2: "<END>",
    3: "<UNKNOWN>",
  };
  const intToTokenLangTwo = {
    0: "<PAD>",
    1: "<START>",
    2: "<END>",
    3: "<UNKNOWN>",
  };

  Object.keys(vocabEn).forEach((token, index) => {
    tokenToIntEn[token] = index + 4;
    intToTokenLangOne[index + 4] = token;
  });

  Object.keys(vocabPt).forEach((token, index) => {
    tokenToIntPt[token] = index + 4;
    intToTokenLangTwo[index + 4] = token;
  });

  return [
    tokenizedData.map((item) => ({
      [languageOne]: padSequence(
        [1, ...item[languageOne].map((token) => tokenToIntEn[token] || 3), 2],
        max_tokens
      ),
      [languageTwo]: padSequence(
        [1, ...item[languageTwo].map((token) => tokenToIntPt[token] || 3), 2],
        max_tokens
      ),
    })),
    { intToTokenLangOne, intToTokenLangTwo },
  ];
}

function padSequence(sequence, length) {
  while (sequence.length < length) {
    sequence.push(0);
  }
  if (sequence.length > length) {
    let newSeq = sequence.slice(0, length - 1);
    newSeq.push(2);
    return newSeq;
  } else {
    return sequence.slice(0, length);
  }
}

function detokenizeSentence(tokenizedIntegers, intToToken) {
  return tokenizedIntegers.map((int) => intToToken[int] || "").join(" ");
}

function tokenizeSentence(sentence) {
  const tokens = [];
  const doc = nlp.readDoc(sentence);
  doc.tokens().each((token) => tokens.push(token.out()));
  return tokens;
}

function splitData(data) {
  const totalData = data.length;
  const trainData = data.slice(0, Math.floor(totalData * 0.8));
  const validationData = data.slice(
    Math.floor(totalData * 0.8),
    Math.floor(totalData * 0.9)
  );
  const testData = data.slice(Math.floor(totalData * 0.9), totalData);
  return { trainData, validationData, testData };
}

function processJson(
  jsonData,
  languageOne,
  languageTwo,
  maxVocabEn = 5000,
  maxVocabPt = 5000,
  max_tokens = 50
) {
  const tokenizedData = jsonData.map((item) => ({
    [languageOne]: tokenizeSentence(item[languageOne]),
    [languageTwo]: tokenizeSentence(item[languageTwo]),
  }));

  const { vocabularyEn, vocabularyPt } = buildVocabulary(
    tokenizedData,
    languageOne,
    languageTwo,
    maxVocabEn,
    maxVocabPt
  );
  const [dataWithIntegers, toks] = convertTokensToIntegers(
    tokenizedData,
    vocabularyEn,
    vocabularyPt,
    languageOne,
    languageTwo,
    max_tokens
  );
  const { trainData, validationData, testData } = splitData(dataWithIntegers);

  const tokenizers = {
    [languageOne]: toks.intToTokenLangOne,
    [languageTwo]: toks.intToTokenLangTwo,
  };

  return {
    trainData,
    validationData,
    testData,
    tokenizers,
  };
}

function shiftTokens(targetLanguage, endToken = 2) {
  const train_target = [];
  const label_target = [];

  for (const sentence of targetLanguage) {
    // Create new arrays to avoid mutation of the original arrays
    const newTrainSentence = [...sentence];
    const newLabelSentence = [...sentence];

    // For train_target, remove the element with value of endToken
    const index = newTrainSentence.indexOf(endToken);
    if (index !== -1) {
      newTrainSentence.splice(index, 1);
    }
    newTrainSentence.push(0); // Add 0 to the end

    // For label_target, remove the last element
    newLabelSentence.shift();
    newLabelSentence.push(0); // Add 0 to the end

    train_target.push(newTrainSentence);
    label_target.push(newLabelSentence);
  }

  return [train_target, label_target];
}

// Takes in a sentence and returns a padded list of tokens of max_length with a start and end symbol based on the intToToken dictionary provided

function wordsToIntTokens(sentence, intToToken, max_length = 10) {
  const tokens = tokenizeSentence(sentence);

  // Switches the keys and values of the intToToken dictionary
  const tokenToInt = Object.keys(intToToken).reduce((acc, key) => {
    acc[intToToken[key]] = key;
    return acc;
  }, {});

  // Converts the tokens to integers based on the tokenToInt dictionary and adds the start and end token

  const tokenizedIntegers = [
    1,
    ...tokens.map((token) => tokenToInt[token] || 3),
    2,
  ];

  // Pads the tokenizedIntegers to max_length or truncates it to max_length
  return padSequence(tokenizedIntegers, max_length);
}

export { processJson, detokenizeSentence, shiftTokens, wordsToIntTokens };
