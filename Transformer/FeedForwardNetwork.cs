using System;
using Transformer.Utils;

namespace Transformer
{
    public class FeedForwardNetwork
    {
        private Tensor W1, W2;
        private Tensor b1, b2;

        private Optimizer W1Optimizer, W2Optimizer, b1Optimizer, b2Optimizer;

        public FeedForwardNetwork(int dff, int embeddingSize)
        {
            W1 = new Tensor(embeddingSize, dff);
            W2 = new Tensor(dff, embeddingSize);
            b1 = new Tensor(dff);
            b2 = new Tensor(embeddingSize);
            GenerateRandomLayers();

            W1Optimizer = new Optimizer(W1);
            W2Optimizer = new Optimizer(W2);
            b1Optimizer = new Optimizer(b1);
            b2Optimizer = new Optimizer(b2);
        }

        public Tensor FeedForward(Tensor G)
        {
            // First layer
            Tensor FFN1 = Tensor.MatMul(G, W1).VecAdd(b1);
            FFN1.ReLU();

            // Second layer
            Tensor FFN2 = Tensor.MatMul(FFN1, W2).VecAdd(b2);

            return FFN2;
        }

        private void GenerateRandomLayers()
        {
            W1.GenerateNormalRandomValues();
            W2.GenerateNormalRandomValues();
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            W1Optimizer.MakeTrainingStep(learningRate, step, W1);
            W2Optimizer.MakeTrainingStep(learningRate, step, W2);
            b1Optimizer.MakeTrainingStep(learningRate, step, b1);
            b2Optimizer.MakeTrainingStep(learningRate, step, b2);
        }

    }
}
