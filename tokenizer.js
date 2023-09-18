let vocabularyEn = { "<PAD>": 0, "<START>": 1, "<END>": 2 };
let vocabularyPt = { "<PAD>": 0, "<START>": 1, "<END>": 2 };
let intToTokenEn = { 0: "<PAD>", 1: "<START>", 2: "<END>" };
let intToTokenPt = { 0: "<PAD>", 1: "<START>", 2: "<END>" };

function buildVocabulary(tokenizedData) {
  tokenizedData.forEach((item) => {
    item.en.forEach((token) => {
      vocabularyEn[token] = (vocabularyEn[token] || 0) + 1;
    });
    item.pt.forEach((token) => {
      vocabularyPt[token] = (vocabularyPt[token] || 0) + 1;
    });
  });
}

function convertTokensToIntegers(tokenizedData, max_tokens = 50) {
  const tokenToIntEn = {};
  const tokenToIntPt = {};
  const vocabArrayEn = Object.keys(vocabularyEn);
  const vocabArrayPt = Object.keys(vocabularyPt);

  vocabArrayEn.forEach((token, index) => {
    tokenToIntEn[token] = index;
    intToTokenEn[index] = token;
  });

  vocabArrayPt.forEach((token, index) => {
    tokenToIntPt[token] = index;
    intToTokenPt[index] = token;
  });

  return tokenizedData.map((item) => ({
    en: padSequence(
      [tokenToIntEn["<START>"], ...item.en.map((token) => tokenToIntEn[token] || tokenToIntEn["<PAD>"]), tokenToIntEn["<END>"]],
      max_tokens
    ),
    pt: padSequence(
      [tokenToIntPt["<START>"], ...item.pt.map((token) => tokenToIntPt[token] || tokenToIntPt["<PAD>"]), tokenToIntPt["<END>"]],
      max_tokens
    ),
  }));
}

function padSequence(sequence, length) {
  while (sequence.length < length) {
    sequence.push(vocabularyEn["<PAD>"]);
  }
  return sequence.slice(0, length);
}

function detokenizeSentence(tokenizedIntegers, intToToken) {
  return tokenizedIntegers.map((int) => intToToken[int] || "").join(" ");
}


function tokenizeSentence(sentence) {
  // Unicode normalization
  sentence = sentence.normalize("NFD").replace(/[\u0300-\u036f]/g, "");

  // Case normalization
  sentence = sentence.toLowerCase();

  // Handling apostrophes
  sentence = sentence
    .replace(/n't/g, " not")
    .replace(/'ve/g, " have")
    .replace(/'re/g, " are")
    .replace(/'ll/g, " will")
    .replace(/'d/g, " would")
    .replace(/'s/g, " is");

  // Handling numbers (replacing with a placeholder)
  sentence = sentence.replace(/\b\d+\b/g, "<NUM>");

  // Handling URLs (replacing with a placeholder)
  sentence = sentence.replace(/https?:\/\/[^\s]+/g, "<URL>");

  // Handling email addresses (replacing with a placeholder)
  sentence = sentence.replace(
    /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
    "<EMAIL>"
  );

  // Tokenization while keeping hyphenated words intact
  return sentence
    .replace(/[.,!?;:]/g, " $& ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ");
}

function tokenizeSentencePt(sentence) {
  // Unicode normalization (keeping accented characters as they are important in Portuguese)
  sentence = sentence.normalize("NFC");

  // Case normalization
  sentence = sentence.toLowerCase();

  // Handling contractions specific to Portuguese
  sentence = sentence
    .replace(/ à /g, " a")
    .replace(/ ao /g, " a")
    .replace(/ pelos /g, " por os")
    .replace(/ pelas /g, " por as")
    .replace(/ do /g, " de o")
    .replace(/ da /g, " de a")
    .replace(/ dos /g, " de os")
    .replace(/ das /g, " de as");

  // Handling numbers (replacing with a placeholder)
  sentence = sentence.replace(/\b\d+\b/g, "<NUM>");

  // Handling URLs (replacing with a placeholder)
  sentence = sentence.replace(/https?:\/\/[^\s]+/g, "<URL>");

  // Handling email addresses (replacing with a placeholder)
  sentence = sentence.replace(
    /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
    "<EMAIL>"
  );

  // Tokenization while keeping hyphenated words and accented characters intact
  return sentence
    .replace(/[.,!?;:]/g, " $& ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ");
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

function processJson(jsonData, max_tokens = 50) {
    const tokenizedData = jsonData.map((item) => ({
      en: tokenizeSentence(item.en),
      pt: tokenizeSentencePt(item.pt),
    }));
  
    buildVocabulary(tokenizedData);
    const dataWithIntegers = convertTokensToIntegers(tokenizedData, max_tokens);
    const { trainData, validationData, testData } = splitData(dataWithIntegers);
  
    return {
      trainData,
      validationData,
      testData,
      tokenizers: {
        en: intToTokenEn,
        pt: intToTokenPt,
      },
    };
  }
  

export { processJson, detokenizeSentence };
