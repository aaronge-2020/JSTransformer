const winkNLP = (await import('https://cdn.skypack.dev/wink-nlp')).default;
const mod = (await import('https://cdn.skypack.dev/wink-eng-lite-web-model')).default;
const nlp = winkNLP( mod );

let vocabularyEn = { "<PAD>": 0, "<START>": 1, "<END>": 2 };
let vocabularyPt = { "<PAD>": 0, "<START>": 1, "<END>": 2 };
let intToTokenLangOne = { 0: "<PAD>", 1: "<START>", 2: "<END>" };
let intToTokenLangTwo = { 0: "<PAD>", 1: "<START>", 2: "<END>" };

function buildVocabulary(tokenizedData, languageOne, languageTwo, maxVocabEn, maxVocabPt) {
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
  vocabularyEn = Object.entries(vocabularyEn).sort(([,a],[,b]) => b-a).slice(0, maxVocabEn).reduce((acc, [k, v]) => ({ ...acc, [k]: v }), {});
  vocabularyPt = Object.entries(vocabularyPt).sort(([,a],[,b]) => b-a).slice(0, maxVocabPt).reduce((acc, [k, v]) => ({ ...acc, [k]: v }), {});

  return { vocabularyEn, vocabularyPt };
}

function convertTokensToIntegers(tokenizedData, vocabEn, vocabPt, languageOne, languageTwo, max_tokens = 50) {
  const tokenToIntEn = { "<UNKNOWN>": 3 };
  const tokenToIntPt = { "<UNKNOWN>": 3 };
  const intToTokenLangOne = { 0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNKNOWN>" };
  const intToTokenLangTwo = { 0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNKNOWN>" };

  Object.keys(vocabEn).forEach((token, index) => {
    tokenToIntEn[token] = index + 4;
    intToTokenLangOne[index + 4] = token;
  });

  Object.keys(vocabPt).forEach((token, index) => {
    tokenToIntPt[token] = index + 4;
    intToTokenLangTwo[index + 4] = token;
  });

  return [tokenizedData.map((item) => ({
    [languageOne]: padSequence(
      [1, ...item[languageOne].map((token) => tokenToIntEn[token] || 3), 2],
      max_tokens
    ),
    [languageTwo]: padSequence(
      [1, ...item[languageTwo].map((token) => tokenToIntPt[token] || 3), 2],
      max_tokens
    ),
  })), { intToTokenLangOne, intToTokenLangTwo }];
}

function padSequence(sequence, length) {
  while (sequence.length < length) {
    sequence.push(0);
  }
  return sequence.slice(0, length);
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

function processJson(jsonData, languageOne, languageTwo, maxVocabEn = 5000, maxVocabPt = 5000, max_tokens = 50) {
  const tokenizedData = jsonData.map((item) => ({
    [languageOne]: tokenizeSentence(item[languageOne]),
    [languageTwo]: tokenizeSentence(item[languageTwo]),
  }));

  const { vocabularyEn, vocabularyPt } = buildVocabulary(tokenizedData, languageOne, languageTwo, maxVocabEn, maxVocabPt);
  const [dataWithIntegers, toks] = convertTokensToIntegers(tokenizedData, vocabularyEn, vocabularyPt, languageOne, languageTwo, max_tokens);
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
  // Create a deep copy of targetLanguage for label_target
  let label_target = JSON.parse(JSON.stringify(targetLanguage));

  // Remove the last element from each sentence in label_target
  label_target = label_target.map((sent) => {
    return sent.slice(1,sent.length);
  });

  // Add 0 to the end of each sentence in label_target
  label_target.forEach((sent) => sent.push(0));

  // Create a deep copy of targetLanguage for label_target
  let train_target = JSON.parse(JSON.stringify(targetLanguage));

  // Remove the element with value of endToken from each sentence in train_target
  train_target.forEach((sentence) => {
    const index = sentence.indexOf(endToken);
    if (index !== -1) {
      sentence.splice(index, 1);
    }
  });

  // Add 0 to the end of each sentence in label_target
  train_target.forEach((sentence) => sentence.push(0));

  return [train_target, label_target];
}


export { processJson, detokenizeSentence, shiftTokens };
