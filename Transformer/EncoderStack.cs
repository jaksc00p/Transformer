using System;
using Transformer.Utils;

namespace Transformer
{
    public class EncoderStack
    {
        private int Nx;
        private List<EncoderLayer> encoderLayers = new List<EncoderLayer>();

        public EncoderStack(int Nx, int embeddingSize, int dk, int dv, int h, int dff)
        {
            this.Nx = Nx;

            for (int i = 0; i < Nx; i++)
            {
                encoderLayers.Add(new EncoderLayer(embeddingSize, dk, dv, h, dff));
            }
        }

        public Tensor Encode(Tensor word_embeddings, bool useDropout)
        {
            var encoderOutput = encoderLayers[0].Encode(word_embeddings, useDropout);
            for (int i = 1; i < Nx; i++)
            {
                encoderOutput = encoderLayers[i].Encode(encoderOutput, useDropout);
            }

            return encoderOutput;
        }

        public void SetDropoutNodes(double dropout)
        {
            for (int i = 0; i < Nx; i++)
            {
                encoderLayers[i].SetDropoutNodes(dropout);
            }
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            for (int i = 0; i < Nx; i++)
            {
                encoderLayers[i].MakeTrainingStep(learningRate, step);
            }
        }

    }
}
