using System;

namespace Transformer
{
    public sealed class RandomNumbers
    {
        private static readonly RandomNumbers instance = new RandomNumbers();
        public static RandomNumbers Instance
        {
            get
            {
                return instance;
            }
        }

        // Explicit static constructor to tell C# compiler not to mark type as beforefieldinit
        static RandomNumbers() { }

        private RandomNumbers() { }

        // private Random _rand = new Random(DateTime.Now.Second);
        private Random _rand = new Random(0);

        public double GetNextUniformNumber()
        {
            return _rand.NextDouble();
        }

        public double GetNextNormalNumber()
        {
            double u1 = 1.0 - _rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - _rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)

            return randStdNormal;
        }
    }
}
