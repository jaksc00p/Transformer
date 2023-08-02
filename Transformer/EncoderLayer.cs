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

        public EncoderLayer(int embeddingSize, int dk, int dv, int h, int dff)
        {
            this.embeddingSize = embeddingSize;

            mha = new MultiHeadAttention(dk, dv, h, embeddingSize, false);
            ff = new FeedForwardNetwork(dff, embeddingSize);

            dropoutVector1 = new bool[embeddingSize];
            dropoutVector2 = new bool[embeddingSize];
        }

        public Tensor Encode(Tensor encoderInput, bool useDropout)
        {
            // Multi headed attention
            var attentionFilteredData = mha.Update(encoderInput);
            if (useDropout)
                attentionFilteredData.Dropout(dropoutVector1);
            attentionFilteredData = Tensor.AddNorm(encoderInput, attentionFilteredData);

            // Feed forward neural network
            var feedForwardOutput = ff.FeedForward(attentionFilteredData);
            if (useDropout)
                feedForwardOutput.Dropout(dropoutVector2);
            feedForwardOutput = Tensor.AddNorm(attentionFilteredData, feedForwardOutput);

            return feedForwardOutput;
        }

        public void SetDropoutNodes(double dropout)
        {
            for (int i = 0; i < embeddingSize; i++)
            {
                dropoutVector1[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropout)
                    dropoutVector1[i] = true;

                dropoutVector2[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropout)
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
