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

        private bool[] dropoutVector1, dropoutVector2, dropoutVector3;

        public DecoderLayer(int embeddingSize, int dk, int dv, int h, int dff)
        {
            this.embeddingSize = embeddingSize;

            mha = new MultiHeadAttention(dk, dv, h, embeddingSize, false);
            mha_masked = new MultiHeadAttention(dk, dv, h, embeddingSize, true);
            ff = new FeedForwardNetwork(dff, embeddingSize);

            dropoutVector1 = new bool[embeddingSize];
            dropoutVector2 = new bool[embeddingSize];
            dropoutVector3 = new bool[embeddingSize];
        }

        public Tensor Decode(Tensor encoderOutput, Tensor decoderInput, bool useDropout)
        {
            // Masked multi headed attention
            var maskedAttentionFilteredData = mha_masked.Update(decoderInput);
            if (useDropout)
                maskedAttentionFilteredData.Dropout(dropoutVector1);
            maskedAttentionFilteredData = Tensor.AddNorm(decoderInput, maskedAttentionFilteredData);

            // Multi headed attention
            var attentionFilteredData = mha.Update(encoderOutput, maskedAttentionFilteredData);
            if (useDropout)
                attentionFilteredData.Dropout(dropoutVector2);
            attentionFilteredData = Tensor.AddNorm(maskedAttentionFilteredData, attentionFilteredData);

            // Feed forward neural network
            var feedForwardOutput = ff.FeedForward(attentionFilteredData);
            if (useDropout)
                feedForwardOutput.Dropout(dropoutVector3);
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

                dropoutVector3[i] = false;
                if (RandomNumbers.Instance.GetNextUniformNumber() < dropout)
                    dropoutVector3[i] = true;
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
