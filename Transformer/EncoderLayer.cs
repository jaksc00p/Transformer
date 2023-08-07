using System;
using Transformer.Utils;

namespace Transformer
{
    public class EncoderLayer
    {
        private int embeddingSize;

        private MultiHeadAttention mha;
        private FeedForwardNetwork ff;

        private bool[] dropoutMask1, dropoutMask2;
        private double dropoutRate = 0;

        public EncoderLayer(int embeddingSize, int dk, int dv, int h, int dff)
        {
            this.embeddingSize = embeddingSize;

            mha = new MultiHeadAttention(dk, dv, h, embeddingSize, false);
            ff = new FeedForwardNetwork(dff, embeddingSize);

            dropoutMask1 = new bool[embeddingSize];
            dropoutMask2 = new bool[embeddingSize];
        }

        public Tensor Encode(Tensor encoderInput, bool isTraining)
        {
            // Multi headed attention
            var attentionFilteredData = mha.Update(encoderInput);
            if (isTraining && dropoutRate > 0)
                attentionFilteredData = attentionFilteredData.Dropout(dropoutMask1, dropoutRate);
            attentionFilteredData = Tensor.AddNorm(encoderInput, attentionFilteredData);

            // Feed forward neural network
            var feedForwardOutput = ff.FeedForward(attentionFilteredData);
            if (isTraining && dropoutRate > 0)
                feedForwardOutput = feedForwardOutput.Dropout(dropoutMask2, dropoutRate);
            feedForwardOutput = Tensor.AddNorm(attentionFilteredData, feedForwardOutput);

            return feedForwardOutput;
        }

        public void SetDropoutNodes(double dropoutRate)
        {
            if (dropoutRate < 0 || dropoutRate >= 1)
                throw new ArgumentException("Error: dropout rate must be >= 0 and < 1");

            this.dropoutRate = dropoutRate;

            for (int i = 0; i < embeddingSize; i++)
            {
                dropoutMask1[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropoutRate)
                    dropoutMask1[i] = true;

                dropoutMask2[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropoutRate)
                    dropoutMask2[i] = true;
            }
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            mha.MakeTrainingStep(learningRate, step);
            ff.MakeTrainingStep(learningRate, step);
        }

    }
}
