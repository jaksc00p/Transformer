using System;
using Transformer.Utils;

namespace Transformer
{
    public class DecoderLayer
    {
        private int embeddingSize;

        private MultiHeadAttention mha;
        private MultiHeadAttention mha_masked;
        public FeedForwardNetwork ff;

        private bool[] dropoutMask1, dropoutMask2, dropoutMask3;
        private double dropoutRate = 0;

        public DecoderLayer(int embeddingSize, int dk, int dv, int h, int dff)
        {
            this.embeddingSize = embeddingSize;

            mha = new MultiHeadAttention(dk, dv, h, embeddingSize, false);
            mha_masked = new MultiHeadAttention(dk, dv, h, embeddingSize, true);
            ff = new FeedForwardNetwork(dff, embeddingSize);

            dropoutMask1 = new bool[embeddingSize];
            dropoutMask2 = new bool[embeddingSize];
            dropoutMask3 = new bool[embeddingSize];
        }

        public Tensor Decode(Tensor encoderOutput, Tensor decoderInput, bool isTraining)
        {
            // Masked multi headed attention
            var maskedAttentionFilteredData = mha_masked.Update(decoderInput);
            if (isTraining && dropoutRate > 0)
                maskedAttentionFilteredData = maskedAttentionFilteredData.Dropout(dropoutMask1, dropoutRate);
            maskedAttentionFilteredData = Tensor.AddNorm(decoderInput, maskedAttentionFilteredData);

            // Multi headed attention
            var attentionFilteredData = mha.Update(encoderOutput, maskedAttentionFilteredData);
            if (isTraining && dropoutRate > 0)
                attentionFilteredData = attentionFilteredData.Dropout(dropoutMask2, dropoutRate);
            attentionFilteredData = Tensor.AddNorm(maskedAttentionFilteredData, attentionFilteredData);

            // Feed forward neural network
            var feedForwardOutput = ff.FeedForward(attentionFilteredData);
            if (isTraining && dropoutRate > 0)
                feedForwardOutput = feedForwardOutput.Dropout(dropoutMask3, dropoutRate);
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

                dropoutMask3[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropoutRate)
                    dropoutMask3[i] = true;
            }
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            mha_masked.MakeTrainingStep(learningRate, step);
            mha.MakeTrainingStep(learningRate, step);
            ff.MakeTrainingStep(learningRate, step);
        }

    }
}
