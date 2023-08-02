using System;

namespace Transformer
{
    public static class TextProcessing
    {
        public static void Load(string filename, int nrSentences, out List<List<string>> englishSentences, out List<List<string>> spanishSentences)
        {
            englishSentences = new List<List<string>>();
            spanishSentences = new List<List<string>>();

            using (var reader = new StreamReader(filename))
            {
                if (reader != null)
                {
                    int s = 1;
                    char[] splitchars = new char[] { '.', '?', '!', '\t' };
                    string line = reader.ReadLine();
                    while (line != null)
                    {
                        line = line.ToLower();
                        bool isquestion = line.Contains("?");
                        bool isexclamation = line.Contains("!");
                        string[] split = line.Split(splitchars, StringSplitOptions.RemoveEmptyEntries);
                        if (isquestion)
                        {
                            split[0] += " ?";
                            split[1] += " ?";
                        }
                        if (isexclamation)
                        {
                            split[0] += " !";
                            split[1] += " !";
                        }
                        englishSentences.Add(new List<string>(split[0].Split()));
                        spanishSentences.Add(new List<string>(split[1].Split()));
                        if (s++ >= nrSentences)
                            break;

                        line = reader.ReadLine();
                    }
                }
            }
        }

        public static List<List<string>> ProcessSentence(string sentenceString)
        {
            var sentence = new List<List<string>>();

            sentenceString = sentenceString.Replace("?", " ?");
            sentenceString = sentenceString.Replace("!", " !");
            sentenceString = sentenceString.Replace(".", "");

            char[] splitchars = new char[] { ' ', '\t' };
            sentence.Add(new List<string>(sentenceString.Split(splitchars, StringSplitOptions.RemoveEmptyEntries)));

            return sentence;
        }

        public static string ProcessSentence(List<List<string>> sentence, int s)
        {
            if (sentence.Count <= s)
                throw new ArgumentException("Index out of range");

            string sentenceString = "";
            for (int i = 0; i < sentence[s].Count; i++)
            {
                sentenceString += sentence[s][i].ToString() + " ";
            }

            return sentenceString.Trim();
        }

        public static int CalculateSequenceLength(List<List<string>> englishSentences, List<List<string>> spanishSentences)
        {
            int sequenceLength = 0;
            for (int s = 0; s < englishSentences.Count; s++)
            {
                sequenceLength = Math.Max(sequenceLength, englishSentences[s].Count());
                sequenceLength = Math.Max(sequenceLength, spanishSentences[s].Count());
            }

            return sequenceLength;
        }

        public static int CalculateMaxSentenceLength(List<List<string>> sentences)
        {
            int maxSentenceLength = 0;
            for (int s = 0; s < sentences.Count; s++)
            {
                maxSentenceLength = Math.Max(maxSentenceLength, sentences[s].Count());
            }

            return maxSentenceLength;
        }

        public static void InsertStartAndStopCharacters(List<List<string>> correctSpanishSentences)
        {
            for (int s = 0; s < correctSpanishSentences.Count; s++)
            {
                if (correctSpanishSentences[s][0] != "<")
                    correctSpanishSentences[s].Insert(0, "<");
                if (correctSpanishSentences[s][correctSpanishSentences[s].Count - 1] != ">")
                    correctSpanishSentences[s].Add(">");
            }
        }

        public static List<List<string>> InitializeSpanishSentences(int batchSize)
        {
            List<List<string>> translatedSpanishSentences = new List<List<string>>();
            for (int s = 0; s < batchSize; s++)
            {
                translatedSpanishSentences.Add(new List<string>() { "<" });
            }

            return translatedSpanishSentences;
        }

        public static void AddWordsToSentences(int batchSize, bool isTraining, string[] spanishWords,
            List<List<string>> translatedSpanishSentences)
        {
            for (int s = 0; s < batchSize; s++)
            {
                int sentenceLength = translatedSpanishSentences[s].Count;
                if (sentenceLength > 0 && translatedSpanishSentences[s][sentenceLength - 1] == ">")
                    continue;

                translatedSpanishSentences[s].Add(spanishWords[s]);
            }
        }

    }
}
