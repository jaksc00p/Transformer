using System;
using Transformer.Utils;

namespace Transformer
{
    /// <summary>
    /// Use a learned embedding layer to reduce the size of the word embedding space.
    /// </summary>
    public class Embedding
    {
        private List<string> allWords = new List<string>();
        private Dictionary<string, int> one_hot = new Dictionary<string, int>();
        private bool[] dropoutVector;

        // Learned linear embedding layer
        private Tensor embeddingLayer;

        private Optimizer embeddingLayerOptimizer;

        public int DictionarySize { get { return one_hot.Count; } }
        public int EmbeddingSize { get; private set; }
        public int SequenceLength { get; private set; }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="embeddingSize"></param>
        /// <param name="sequenceLength"></param>
        /// <param name="sentences"></param>
        public Embedding(int embeddingSize, int sequenceLength, List<List<string>> sentences)
        {
            EmbeddingSize = embeddingSize;
            SequenceLength = sequenceLength;

            OneHotEmbedding(sentences);
            embeddingLayer = new Tensor(DictionarySize, EmbeddingSize);
            embeddingLayer.GenerateNormalRandomValues();

            embeddingLayerOptimizer = new Optimizer(embeddingLayer);

            dropoutVector = new bool[embeddingSize];
        }

        /// <summary>
        /// Multiply the one-hot embeddings with the embedding layer to project onto a smaller space
        /// </summary>
        /// <param name="sentences"></param>
        /// <param name="isTraining"></param>
        /// <returns></returns>
        public Tensor Embed(List<List<string>> sentences, bool useDropout)
        {
            int batchSize = sentences.Count;
            Tensor wordEmbeddings = new Tensor(batchSize, SequenceLength, EmbeddingSize);

            int s = 0;
            foreach (List<string> sentence in sentences)
            {
                int word_count = 0;
                foreach (string word in sentence)
                {
                    // No need for matrix multiplication since only one element of vector is nonzero
                    int pos = one_hot[word.ToLower()];
                    for (int i = 0; i < EmbeddingSize; i++)
                    {
                        wordEmbeddings[s, word_count, i] = embeddingLayer[pos, i];
                    }
                    word_count++;
                }

                AddPositionalEncoding(wordEmbeddings, s, sentence.Count());
                s++;
            }

            if (useDropout)
                wordEmbeddings.Dropout(dropoutVector);

            return wordEmbeddings;
        }

        /// <summary>
        /// Cross entropy loss function between the correct word in a sentence and the decoder output word.
        /// For a batch of several sentences the loss is accumulated.
        /// </summary>
        /// <param name="filteredOutout"></param>
        /// <param name="correctSpanishSentences"></param>
        /// <param name="w"></param>
        /// <param name="loss"></param>
        public void CalculateLossFunction(Tensor filteredOutout, List<List<string>> correctSpanishSentences, int w, ref Rev loss)
        {
            for (int s = 0; s < correctSpanishSentences.Count(); s++)
            {
                if (w >= correctSpanishSentences[s].Count())
                    continue;

                string correctWord = correctSpanishSentences[s][w];

                int ind = GetWordIndex(correctWord);
                loss -= filteredOutout[s, 0, ind].Log();
            }
        }

        /// <summary>
        /// Get a word based on its index in the dictionary
        /// </summary>
        /// <param name="indexes"></param>
        /// <returns></returns>
        public string[] GetWords(int[] indexes)
        {
            string[] words = new string[indexes.Length];
            for (int s = 0; s < indexes.Length; s++)
            {
                words[s] = allWords[indexes[s]];
            }

            return words;
        }

        /// <summary>
        /// Get the index of a specific word in a dictionary
        /// </summary>
        /// <param name="word"></param>
        /// <returns></returns>
        public int GetWordIndex(string word)
        {
            return one_hot[word];
        }

        public bool AllWordsInDictionary(List<List<string>> sentences, out string wordNotInDictionary)
        {
            wordNotInDictionary = "";

            foreach (List<string> sentence in sentences)
            {
                foreach (string w in sentence)
                {
                    if (!one_hot.ContainsKey(w))
                    {
                        wordNotInDictionary = w;
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Encode all words in a dictionary with one-hot embedding
        /// </summary>
        /// <param name="sentences"></param>
        private void OneHotEmbedding(List<List<string>> sentences)
        {
            int word_index = 0;
            foreach (List<string> sentence in sentences)
            {
                foreach (string word in sentence)
                {
                    if (!one_hot.ContainsKey(word.ToLower()))
                    {
                        allWords.Add(word.ToLower());
                        one_hot.Add(word.ToLower(), word_index++);
                    }
                }
            }
        }

        /// <summary>
        /// Add positional encoding to embedded words according to "Attention is all you need"
        /// </summary>
        /// <param name="wordEmbeddings"></param>
        /// <param name="s"></param>
        /// <param name="sentenceLength"></param>
        private void AddPositionalEncoding(Tensor wordEmbeddings, int s, int sentenceLength)
        {
            for (int pos = 0; pos < sentenceLength; pos++)
            {
                for (int i = 0; i < EmbeddingSize; i++)
                {
                    double pe;
                    if (i % 2 == 0)
                    {
                        pe = Math.Sin(pos / Math.Pow(10000, i / (double)EmbeddingSize));
                    }
                    else
                    {
                        pe = Math.Cos(pos / Math.Pow(10000, (i - 1) / (double)EmbeddingSize));
                    }
                    wordEmbeddings[s, pos, i] += pe;
                }
            }
        }

        public void SetDropoutNodes(double dropout)
        {
            for (int i = 0; i < EmbeddingSize; i++)
            {
                dropoutVector[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropout)
                    dropoutVector[i] = true;
            }
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            embeddingLayerOptimizer.MakeTrainingStep(learningRate, step, embeddingLayer);
        }

    }
}
