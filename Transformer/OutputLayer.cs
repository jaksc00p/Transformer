using System;
using Transformer.Utils;

namespace Transformer
{
    /// <summary>
    /// Produce a flat array with the same dimension as the number of words in the dictionary
    /// </summary>
    public class OutputLayer
    {
        public Tensor Wo;

        private Optimizer WoOptimizer;

        public OutputLayer(int sequenceLength, int embeddingSize, int dictionarySize)
        {
            Wo = new Tensor(embeddingSize * sequenceLength, dictionarySize);
            Wo.GenerateNormalRandomValues();

            WoOptimizer = new Optimizer(Wo);
        }

        public Tensor Output(Tensor input)
        {
            var flatInput = input.Flatten();
            var filteredOutput = Tensor.MatMul(flatInput, Wo); 
            var softmaxOutput = filteredOutput.Softmax();

            return softmaxOutput;
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            WoOptimizer.MakeTrainingStep(learningRate, step, Wo);
        }

    }
}
