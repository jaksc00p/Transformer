using Transformer;

// Load data
int nrSentences = 100;
TextProcessing.Load("TrainingData\\english-spanish.txt", nrSentences, out var allEnglishSentences, out var allSpanishSentences);

// Transformer setup
int batchSize = 10;
int embeddingSize = 8;
int dk = 4;
int dv = 4;
int h = 2;
int dff = 16;
int Nx = 2;
double dropout = 0.0;
TransformerModel transformer = new TransformerModel(Nx, embeddingSize, dk, dv, h, dff,
    batchSize, dropout, allEnglishSentences, allSpanishSentences);

// Training
int nrEpochs = 2;
int nrTrainingSteps = 100;
double learningRate = 0.01;
try
{
    transformer.Train(nrEpochs, nrTrainingSteps, learningRate, batchSize,
        allEnglishSentences, allSpanishSentences);
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

// Testing
try
{
    transformer.Infer();
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

Console.WriteLine("");

