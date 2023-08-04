using System;
using Transformer.Utils;

namespace Transformer
{
    /// <summary>
    /// Transformer architecture as described in "Attention is all you need"
    /// </summary>
    public class TransformerModel
    {
        private int sequenceLength;
        private double dropout;

        private EncoderStack encoder;
        private DecoderStack decoder;
        private Embedding englishEmbedding;
        private Embedding spanishEmbedding;
        private OutputLayer outputLayer;

        private Rev loss;

        public TransformerModel(int Nx, int embeddingSize, int dk, int dv, int h, int dff, int batchSize, double dropout,
            List<List<string>> allEnglishSentences, List<List<string>> allSpanishSentences)
        {
            this.dropout = dropout;

            TextProcessing.InsertStartAndStopCharacters(allSpanishSentences);
            sequenceLength = TextProcessing.CalculateSequenceLength(allEnglishSentences, allSpanishSentences);

            englishEmbedding = new Embedding(embeddingSize, sequenceLength, allEnglishSentences);
            spanishEmbedding = new Embedding(embeddingSize, sequenceLength, allSpanishSentences);
            encoder = new EncoderStack(Nx, embeddingSize, dk, dv, h, dff);
            decoder = new DecoderStack(Nx, embeddingSize, dk, dv, h, dff);
            outputLayer = new OutputLayer(sequenceLength, embeddingSize, spanishEmbedding.DictionarySize);
        }

        public void Train(int nrEpochs, int nrTrainingSteps, double learningRate, int batchSize,
            List<List<string>> allEnglishSentences, List<List<string>> allSpanishSentences)
        {
            if (allEnglishSentences.Count() != allSpanishSentences.Count())
                throw new ArgumentException("Number of sentence pairs must be equal");

            Console.WriteLine("Training:");
            int nrSentences = allSpanishSentences.Count();
            for (int epoch = 1; epoch <= nrEpochs; epoch++)
            {
                for (int b = 0; b < nrSentences / batchSize; b++)
                {
                    Console.WriteLine();
                    Console.WriteLine("Epoch: " + epoch + ", Batch: " + (b + 1));
                    var englishSentences = allEnglishSentences.GetRange(b * batchSize, batchSize);
                    var spanishSentences = allSpanishSentences.GetRange(b * batchSize, batchSize);

                    for (int step = 1; step <= nrTrainingSteps; step++)
                    {
                        SetDropoutNodes();
                        double loss = Translate(batchSize, true, englishSentences, spanishSentences, out _);
                        Console.WriteLine("Step: " + step + ", loss: " + loss.ToString());
                        MakeTrainingStep(learningRate, step);
                    }
                }
            }
        }

        public void Infer()
        {
            Console.WriteLine();
            Console.WriteLine("Inference (type q to quit):");
            Console.WriteLine();

            while (true)
            {
                Console.WriteLine("Write English sentence (max " + sequenceLength + ") words:");
                string line = Console.ReadLine().Trim();
                if (line.ToLower() == "q")
                    return;

                var englishSentence = TextProcessing.ProcessSentence(line);
                if (englishSentence[0].Count > sequenceLength)
                {
                    Console.WriteLine("Sentence too long:");
                    Console.WriteLine();
                    continue;
                }
                if (!englishEmbedding.AllWordsInDictionary(englishSentence, out string wrongWord))
                {
                    Console.WriteLine(wrongWord + " not in dictionary");
                    Console.WriteLine();
                    continue;
                }

                Translate(1, false, englishSentence, null, out List<List<string>> translatedSpanishSentence);
                Console.WriteLine("Spanish translation:");
                string translation = TextProcessing.ProcessSentence(translatedSpanishSentence, 0);
                translation = translation.Replace("<", "");
                translation = translation.Replace(">", "");
                translation = translation.Trim();
                Console.WriteLine(translation);
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Translate a batch of sentenses by generating one word at a time for each sentence with the decoder until max 
        /// length or stopping character.
        /// </summary>
        private double Translate(int batchSize, bool isTraining, List<List<string>> englishSentences,
            List<List<string>> correctSpanishSentences, out List<List<string>> translatedSpanishSentences)
        {
            loss = new Rev(0.0);
            Checkpoints.Instance.ClearCheckpoints();

            var english_word_embeddings = englishEmbedding.Embed(englishSentences, isTraining);
            var encoderOutput = encoder.Encode(english_word_embeddings, isTraining);

            translatedSpanishSentences = TextProcessing.InitializeSpanishSentences(batchSize);
            int nrWords = isTraining ? TextProcessing.CalculateMaxSentenceLength(correctSpanishSentences) : sequenceLength;
            for (int w = 1; w < nrWords; w++)
            {
                var spanish_word_embeddings = spanishEmbedding.Embed(translatedSpanishSentences, isTraining);
                Tensor decoder_output = decoder.Decode(encoderOutput, spanish_word_embeddings, isTraining);
                Tensor output = outputLayer.Output(decoder_output);

                if (isTraining)
                    spanishEmbedding.CalculateLossFunction(output, correctSpanishSentences, w, ref loss);

                int[] spanishWordIndexes = output.GetMaxIndex();
                string[] spanishWords = spanishEmbedding.GetWords(spanishWordIndexes);
                TextProcessing.AddWordsToSentences(batchSize, isTraining, spanishWords, translatedSpanishSentences);
            }

            if (isTraining)
                loss /= (double)(sequenceLength * batchSize);

            return loss;

        }

        private void SetDropoutNodes()
        {
            spanishEmbedding.SetDropoutNodes(dropout);
            englishEmbedding.SetDropoutNodes(dropout);
            encoder.SetDropoutNodes(dropout);
            decoder.SetDropoutNodes(dropout);
        }

        private void CalculateGradient()
        {
            loss.CalculateDerivative(1);
            Checkpoints.Instance.CalculateCheckpointGradients();
        }

        private void MakeTrainingStep(double learningRate, int step)
        {
            CalculateGradient();

            englishEmbedding.MakeTrainingStep(learningRate, step);
            spanishEmbedding.MakeTrainingStep(learningRate, step);
            encoder.MakeTrainingStep(learningRate, step);
            decoder.MakeTrainingStep(learningRate, step);
            outputLayer.MakeTrainingStep(learningRate, step);
        }


    }
}
