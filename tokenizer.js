let vocabularyEn = {};
let vocabularyPt = {};
let intToTokenEn = {};
let intToTokenPt = {};

function buildVocabulary(tokenizedData) {
    tokenizedData.forEach(item => {
        item.en.forEach(token => vocabularyEn[token] = (vocabularyEn[token] || 0) + 1);
        item.pt.forEach(token => vocabularyPt[token] = (vocabularyPt[token] || 0) + 1);
    });
}

function convertTokensToIntegers(tokenizedData) {
    vocabularyEn['<START>'] = 0;
    vocabularyEn['<END>'] = 1;
    vocabularyPt['<START>'] = 0;
    vocabularyPt['<END>'] = 1;

    const vocabArrayEn = Object.keys(vocabularyEn);
    const vocabArrayPt = Object.keys(vocabularyPt);
    const tokenToIntEn = {};
    const tokenToIntPt = {};

    vocabArrayEn.forEach((token, index) => {
        tokenToIntEn[token] = index;  // No need to add 2 now
        intToTokenEn[index] = token;
    });
    
    vocabArrayPt.forEach((token, index) => {
        tokenToIntPt[token] = index;  // No need to add 2 now
        intToTokenPt[index] = token;
    });

    return tokenizedData.map(item => ({
        en: [tokenToIntEn['<START>'], ...item.en.map(token => tokenToIntEn[token]), tokenToIntEn['<END>']],
        pt: [tokenToIntPt['<START>'], ...item.pt.map(token => tokenToIntPt[token]), tokenToIntPt['<END>']]
    }));
}


function detokenizeSentence(tokenizedIntegers, intToToken) {
    return tokenizedIntegers.map(int => intToToken[int] || "").join(' ');
}


function prepareBatches(data, batchSize) {
    const batches = [];
    for(let i = 0; i < data.length; i += batchSize) {
        batches.push(data.slice(i, i + batchSize));
    }
    return batches;
}

function tokenizeSentence(sentence) {
  // Unicode normalization
  sentence = sentence.normalize('NFD').replace(/[\u0300-\u036f]/g, "");

  // Case normalization
  sentence = sentence.toLowerCase();

  // Handling apostrophes
  sentence = sentence.replace(/n't/g, " not").replace(/'ve/g, " have").replace(/'re/g, " are").replace(/'ll/g, " will").replace(/'d/g, " would").replace(/'s/g, " is");

  // Handling numbers (replacing with a placeholder)
  sentence = sentence.replace(/\b\d+\b/g, "<NUM>");

  // Handling URLs (replacing with a placeholder)
  sentence = sentence.replace(/https?:\/\/[^\s]+/g, "<URL>");

  // Handling email addresses (replacing with a placeholder)
  sentence = sentence.replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, "<EMAIL>");

  // Tokenization while keeping hyphenated words intact
  return sentence.replace(/[.,!?;:]/g, ' $& ').replace(/\s+/g, ' ').trim().split(' ');
}

function tokenizeSentencePt(sentence) {
  // Unicode normalization (keeping accented characters as they are important in Portuguese)
  sentence = sentence.normalize('NFC');

  // Case normalization
  sentence = sentence.toLowerCase();

  // Handling contractions specific to Portuguese
  sentence = sentence.replace(/ à /g, " a").replace(/ ao /g, " a").replace(/ pelos /g, " por os").replace(/ pelas /g, " por as").replace(/ do /g, " de o").replace(/ da /g, " de a").replace(/ dos /g, " de os").replace(/ das /g, " de as");

  // Handling numbers (replacing with a placeholder)
  sentence = sentence.replace(/\b\d+\b/g, "<NUM>");

  // Handling URLs (replacing with a placeholder)
  sentence = sentence.replace(/https?:\/\/[^\s]+/g, "<URL>");

  // Handling email addresses (replacing with a placeholder)
  sentence = sentence.replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, "<EMAIL>");

  // Tokenization while keeping hyphenated words and accented characters intact
  return sentence.replace(/[.,!?;:]/g, ' $& ').replace(/\s+/g, ' ').trim().split(' ');
}

function splitData(data) {
  const totalData = data.length;
  const trainData = data.slice(0, Math.floor(totalData * 0.8));
  const validationData = data.slice(Math.floor(totalData * 0.8), Math.floor(totalData * 0.9));
  const testData = data.slice(Math.floor(totalData * 0.9), totalData);
  return { trainData, validationData, testData };
}

function processJson(jsonData) {
    const tokenizedData = jsonData.map(item => ({
        en: tokenizeSentence(item.en),
        pt: tokenizeSentencePt(item.pt)
    }));
  
    buildVocabulary(tokenizedData);
    const dataWithIntegers = convertTokensToIntegers(tokenizedData);
    const { trainData, validationData, testData } = splitData(dataWithIntegers);
  
    const batchSize = 32; // Adjust as needed
    return {
        trainBatches: prepareBatches(trainData, batchSize),
        validationBatches: prepareBatches(validationData, batchSize),
        testBatches: prepareBatches(testData, batchSize),
        tokenizers: {
            en: intToTokenEn,
            pt: intToTokenPt
        }
    };
  }
  
  export{
      processJson,
      detokenizeSentence
  };