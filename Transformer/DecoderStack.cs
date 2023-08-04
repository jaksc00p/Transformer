using System;
using Transformer.Utils;

namespace Transformer
{
    public class DecoderStack
    {
        private int Nx;
        private List<DecoderLayer> decoderLayers = new List<DecoderLayer>();

        public DecoderStack(int Nx, int embeddingSize, int dk, int dv, int h, int dff)
        {
            this.Nx = Nx;

            for (int i = 0; i < Nx; i++)
            {
                decoderLayers.Add(new DecoderLayer(embeddingSize, dk, dv, h, dff));
            }
        }

        public Tensor Decode(Tensor encoderOutput, Tensor word_embeddings, bool isTraining)
        {
            var decoderOutput = decoderLayers[0].Decode(encoderOutput, word_embeddings, isTraining);
            for (int i = 1; i < Nx; i++)
            {
                decoderOutput = decoderLayers[i].Decode(encoderOutput, decoderOutput, isTraining);
            }

            return decoderOutput;
        }

        public void SetDropoutNodes(double dropout)
        {
            for (int i = 0; i < Nx; i++)
            {
                decoderLayers[i].SetDropoutNodes(dropout);
            }
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            for (int i = 0; i < Nx; i++)
            {
                decoderLayers[i].MakeTrainingStep(learningRate, step);
            }
        }

    }
}
