import { toWords } from "https://esm.sh/to-words";
import { cleanWord } from "./clean.ts";

const ALL_WORDS_SPLIT_REGEX = /(?<!\p{L})[a-zāīūṁṃṇṅñṣṭḍḷḥ’']+(?!\p{L})/gimu;
const NUMBER_SPLIT_REGEX = /\b(\d+(?:,\d+)*)\b/gim;

interface TrieNode {
  children: Map<string, TrieNode>;
  key?: string;
}

export class DtSuttaPhonemizer {
  private englishIpaDict: Record<string, string>;
  private paliIpaDict: Record<string, string>;
  private modelSymbolTable: Record<string, number[]>;

  private preProcessRegExes: Array<[RegExp, string]> = [];
  private postProcessRegExes: Array<[RegExp, string]> = [];
  private trie: TrieNode = { children: new Map() };

  private dictionaryResponder?: (dictKey: string, dictUsed: number, dictVal: string) => string;
  private tokenizePhonemes: boolean = false;
  public readonly strictMode: boolean

  constructor(
    englishIpaRef: Record<string, string>,
    paliIpaRef: Record<string, string>,
    modelSymbolTable: Record<string, number[]>,
    dictionaryResponder?: (dictKey: string, dictUsed: number, dictVal: string) => string,
    preProcessRegExFindReplaceList?: string[][] | null,
    postProcessRegExFindReplList?: string[][] | null,
    tokenize: boolean = false,
    strictMode: boolean = false
  ) {
    this.englishIpaDict = englishIpaRef;
    this.paliIpaDict = paliIpaRef;
    this.modelSymbolTable = modelSymbolTable;
    this.dictionaryResponder = dictionaryResponder;
    this.tokenizePhonemes = tokenize;
    this.strictMode = strictMode;

    // 1. Pre-compile Regexes once during construction
    if (preProcessRegExFindReplaceList) {
      this.preProcessRegExes = preProcessRegExFindReplaceList.map(([find, repl, options]) => [
        new RegExp(find, options ?? "gmu"), // "gmiu"
        repl,
      ]);
    }
    if (postProcessRegExFindReplList) {
      this.postProcessRegExes = postProcessRegExFindReplList.map(([find, repl, options]) => [
        new RegExp(find, options ?? "gmiu"),
        repl,
      ]);
    }

    // 2. Build Trie for O(1) phoneme matching during tokenization
    this.buildTrie(Object.keys(this.modelSymbolTable));
  }

  private buildTrie(keys: string[]) {
    for (const key of keys) {
      let node = this.trie;
      for (let i = 0; i < key.length; i++) {
        const char = key[i];
        let child = node.children.get(char);
        if (!child) {
          child = { children: new Map() };
          node.children.set(char, child);
        }
        node = child;
      }
      node.key = key;
    }
  }

  private transformNumberToWords(numStr: string): string {
    return toWords(Number(numStr.replaceAll(",", "")), { localeCode: "en-US" })
      .normalize("NFC")
      .toLowerCase();
  }

  private buildWordListFromText(text: string, wordList: string[], numbersMap: Map<string, string>) {
    const words = new Set<string>();

    // Pass 1: Extract words
    const wordMatches = text.match(ALL_WORDS_SPLIT_REGEX);
    if (wordMatches) {
      for (const match of wordMatches) {
        const cleaned = cleanWord(match) as string;
        if (cleaned) words.add(cleaned);
      }
    }

    // Pass 2: Extract numbers
    const numberMatches = text.match(NUMBER_SPLIT_REGEX);
    if (numberMatches) {
      for (const match of numberMatches) {
        let numberInWords = numbersMap.get(match);
        if (!numberInWords) {
          numberInWords = this.transformNumberToWords(match);
          numbersMap.set(match, numberInWords);
        }
        for (const word of numberInWords.split(" ")) {
          if (word) words.add(word);
        }
      }
    }

    wordList.push(...words);
    wordList.sort((a, b) => b.length - a.length);
  }

  private handleModelRegExReplacements(result: string, regExes: Array<[RegExp, string]>): string {
    for (const [regEx, repl] of regExes) {
      result = result.replace(regEx, repl);
    }
    return result;
  }

  private normaliseNumbers(result: string, numbersMap: Map<string, string>): string {
    if (numbersMap.size === 0) return result;
    return result.replace(NUMBER_SPLIT_REGEX, (match) => numbersMap.get(match) ?? match);
  }

  private cleanUpRedundantPunctuation(result: string): string {
  return result
    .replace(/([.,;:!?])[^\S\r\n]+(?=[.,;:!?])/g, "$1")
    .replace(/([.,;:!?])[.,;:!?]+/g, "$1")
    .replace(/[^\S\r\n]+([.,;:!?])/g, "$1")
    .replace(/([.,;:!?])(?=[\p{L}\p{N}])/gu, "$1 ")
    .replace(/[^\S\r\n]{2,}/g, " ") 
    // Replaced \s with [^\S\r\n] to preserve newlines on blank lines
    .replace(/^[.,;:!?[^\S\r\n]]+/gm, "") 
    // Trims leading/trailing horizontal space per line while keeping line breaks intact
    .split("\n")
    .map(line => line.trim())
    .join("\n");
}

public process(text: string): { phonemes: string; tokens: string[][]; ids: number[][] } {
    const normalizedText = text.normalize("NFC");

    // 1. MUST run pre-processing FIRST so replaced words exist during word extraction
    let normalizedLowerText = this.handleModelRegExReplacements(normalizedText, this.preProcessRegExes);
    normalizedLowerText = normalizedLowerText.toLowerCase();

    const wordList: string[] = [];
    const numbersMap = new Map<string, string>();

    // 2. Extract words from the PRE-PROCESSED text
    this.buildWordListFromText(normalizedLowerText, wordList, numbersMap);

    // 3. Phonemise using the matching wordList
    let ipaText = this.phonemise(normalizedLowerText, wordList, numbersMap);


    // 4. post-processing and cleanup
    ipaText = this.handleModelRegExReplacements(ipaText, this.postProcessRegExes);

    ipaText = this.cleanUpRedundantPunctuation(ipaText);

    // 5. Tokenize to IDs
    return this.tokenize(ipaText);
  }

  private phonemise(text: string, wordList: string[], numbersMap: Map<string, string>): string {
    // Replace numbers
    let result = this.normaliseNumbers(text, numbersMap);

    // Build dictionary map for present words
    const dict: Record<string, string> = {};
    for (const term of wordList) {
      let dictUsed = 0;
      let repl = this.englishIpaDict[term];
      if (repl) {
        dictUsed = 1;
      } else {
        repl = this.paliIpaDict[term];
        if (repl) dictUsed = 2;
      }
      if (this.dictionaryResponder) {
        repl = this.dictionaryResponder(term, dictUsed, repl);
      }
      if (repl) dict[term] = repl;
    }

    const terms = Object.keys(dict);
    if (terms.length > 0) {
      const escapedTerms = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      const pattern = new RegExp(`(?<!\\p{L})(${escapedTerms.join("|")})(?!\\p{L})`, "gimu");
      result = result.replace(pattern, (matched) => dict[matched] ?? matched);
    }

    return result
  }

  private tokenize(phonemes: string): { phonemes: string; tokens: string[][]; ids: number[][] } {
    const tokens: string[][] = [[]];
    const ids: number[][] = [[]];

    phonemes = phonemes.normalize("NFD");

    if (this.tokenizePhonemes) {
      let line = 1;
      let i = 0;

      while (i < phonemes.length) {
        const char = phonemes[i];

        if (char === "\n") {
          tokens.push([]);
          ids.push([]);
          line++;
          i++;
          continue;
        }

        // Trie lookup for optimal longest prefix match
        let node = this.trie;
        let matchedKey: string | null = null;

        for (let j = i; j < phonemes.length; j++) {
          const child = node.children.get(phonemes[j]);
          if (!child) break;
          node = child;
          if (node.key !== undefined) {
            matchedKey = node.key;
          }
        }

        if (matchedKey) {
          tokens[line - 1].push(matchedKey);
          if (Array.isArray(this.modelSymbolTable[matchedKey]))
            ids[line - 1].push(...this.modelSymbolTable[matchedKey]);
          else {
            const symb = this.modelSymbolTable[matchedKey] as unknown;
            ids[line - 1].push((symb as number));
          }
          i += matchedKey.length;
        } else {
          const charCode = phonemes.codePointAt(i)?.toString(16) ?? "unknown";
          const before = phonemes.substring(Math.max(0, i - 3), i);
          const after = phonemes.substring(i + 1, Math.min(phonemes.length, i + 3));
          const err = `L#${line} invalid phoneme \\u${charCode} at pos[${i}], <${before}>${char}<${after}>`;

          if (this.strictMode) throw new Error(err);
          console.error(err);
          i++;
        }
      }
    }

    return { phonemes, tokens, ids };
  }
}