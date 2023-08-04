using System;

namespace Transformer.Utils
{
    /// <summary>
    /// Data type for automatic differentiation with reverse mode accumulation.
    /// </summary>
    public class Rev
    {
        public double Magnitude;
        public double Derivative;

        public Action<double> CalculateDerivative;

        public static implicit operator double(Rev d) => d.Magnitude;

        public static Rev operator +(Rev lhs, Rev rhs) =>
            new Rev(lhs.Magnitude + rhs.Magnitude, dx =>
                {
                    if (dx != 0)
                    {
                        lhs.CalculateDerivative(dx);
                        rhs.CalculateDerivative(dx);
                    }
                });

        public static Rev operator +(Rev lhs, double rhs) =>
            new Rev(lhs.Magnitude + rhs, dx =>
                {
                    if (dx != 0)
                    {
                        lhs.CalculateDerivative(dx);
                    }
                });

        public static Rev operator +(double lhs, Rev rhs) =>
            new Rev(lhs + rhs.Magnitude, dx =>
                {
                    if (dx != 0)
                    {
                        rhs.CalculateDerivative(dx);
                    }
                });

        public static Rev operator -(Rev lhs, Rev rhs) =>
           new Rev(lhs.Magnitude - rhs.Magnitude, dx =>
               {
                   if (dx != 0)
                   {
                       lhs.CalculateDerivative(dx);
                       rhs.CalculateDerivative(-dx);
                   }
               });

        public static Rev operator -(Rev lhs, double rhs) =>
           new Rev(lhs.Magnitude - rhs, dx =>
               {
                   if (dx != 0)
                   {
                       lhs.CalculateDerivative(dx);
                   }
               });

        public static Rev operator -(double lhs, Rev rhs) =>
          new Rev(lhs - rhs.Magnitude, dx =>
              {
                  if (dx != 0)
                  {
                      rhs.CalculateDerivative(-dx);
                  }
              });

        public static Rev operator -(Rev lhs) =>
           new Rev(-lhs.Magnitude,
               dx =>
               {
                   if (dx != 0)
                   {
                       lhs.CalculateDerivative(-dx);
                   }
               });

        public static Rev operator *(Rev lhs, Rev rhs) =>
            new Rev(lhs.Magnitude * rhs.Magnitude,
                dx =>
                {
                    if (dx != 0)
                    {
                        lhs.CalculateDerivative(dx * rhs.Magnitude);
                        rhs.CalculateDerivative(dx * lhs.Magnitude);
                    }
                });

        public static Rev operator *(Rev lhs, double rhs) =>
            new Rev(lhs.Magnitude * rhs,
                dx =>
                {
                    if (dx != 0)
                    {
                        lhs.CalculateDerivative(dx * rhs);
                    }
                });

        public static Rev operator *(double lhs, Rev rhs) =>
            new Rev(lhs * rhs.Magnitude,
                dx =>
                {
                    if (dx != 0)
                    {
                        rhs.CalculateDerivative(dx * lhs);
                    }
                });

        public static Rev operator /(Rev lhs, Rev rhs) =>
            new Rev(lhs.Magnitude / rhs.Magnitude,
                dx =>
                {
                    if (dx != 0)
                    {
                        lhs.CalculateDerivative(dx / rhs.Magnitude);
                        rhs.CalculateDerivative(-dx * lhs.Magnitude / (rhs.Magnitude * rhs.Magnitude));
                    }
                });

        public static Rev operator /(Rev lhs, double rhs) =>
            new Rev(lhs.Magnitude / rhs,
                dx =>
                {
                    if (dx != 0)
                    {
                        lhs.CalculateDerivative(dx / rhs);
                    }
                });

        public static Rev operator /(double lhs, Rev rhs) =>
            new Rev(lhs / rhs.Magnitude,
                dx =>
                {
                    if (dx != 0)
                    {
                        rhs.CalculateDerivative(-dx * lhs / (rhs.Magnitude * rhs.Magnitude));
                    }
                });

        public Rev(double y)
        {
            Magnitude = y;
            Derivative = 0;
            CalculateDerivative = (x) =>
            {
                Derivative += x;
            };
        }

        private Rev(double y, Action<double> dy)
        {
            Magnitude = y;
            Derivative = 0;
            CalculateDerivative = dy;
        }

        public Rev Pow(double e)
        {
            return new Rev(Math.Pow(Magnitude, e),
                dx =>
                {
                    CalculateDerivative(e * Math.Pow(Magnitude, e - 1) * dx);
                });
        }

        public Rev Exp()
        {
            return new Rev(Math.Exp(Magnitude),
                dx =>
                {
                    CalculateDerivative(Math.Exp(Magnitude) * dx);
                });
        }

        public Rev Log()
        {
            return new Rev(Math.Log(Magnitude),
                dx =>
                {
                    CalculateDerivative(1.0 / Magnitude * dx);
                });
        }


    }
}