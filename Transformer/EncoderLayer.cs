using System;
using Transformer.Utils;

namespace Transformer
{
    public class EncoderLayer
    {
        private int embeddingSize;

        private MultiHeadAttention mha;
        private FeedForwardNetwork ff;

        private bool[] dropoutVector1, dropoutVector2;
        private double dropoutRate = 0;

        double DropoutCompensation => 1.0 / (1.0 - dropoutRate);

        public EncoderLayer(int embeddingSize, int dk, int dv, int h, int dff)
        {
            this.embeddingSize = embeddingSize;

            mha = new MultiHeadAttention(dk, dv, h, embeddingSize, false);
            ff = new FeedForwardNetwork(dff, embeddingSize);

            dropoutVector1 = new bool[embeddingSize];
            dropoutVector2 = new bool[embeddingSize];
        }

        public Tensor Encode(Tensor encoderInput, bool isTraining)
        {
            // Multi headed attention
            var attentionFilteredData = mha.Update(encoderInput);
            if (isTraining && dropoutRate > 0)
                attentionFilteredData.Dropout(dropoutVector1);
            if (!isTraining && dropoutRate > 0)
                attentionFilteredData *= DropoutCompensation;
            attentionFilteredData = Tensor.AddNorm(encoderInput, attentionFilteredData);

            // Feed forward neural network
            var feedForwardOutput = ff.FeedForward(attentionFilteredData);
            if (isTraining && dropoutRate > 0)
                feedForwardOutput.Dropout(dropoutVector2);
            if (!isTraining && dropoutRate > 0)
                feedForwardOutput *= DropoutCompensation;
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
                dropoutVector1[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropoutRate)
                    dropoutVector1[i] = true;

                dropoutVector2[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropoutRate)
                    dropoutVector2[i] = true;
            }
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            mha.MakeTrainingStep(learningRate, step);
            ff.MakeTrainingStep(learningRate, step);
        }

    }
}
