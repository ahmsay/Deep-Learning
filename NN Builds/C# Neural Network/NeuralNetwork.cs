using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class NeuralNetwork
    {
        private int[] layers;
        private List<double[,]> weights = new List<double[,]>();
        private List<double[,]> hiddens = new List<double[,]>();
        private List<double[,]> dHiddens = new List<double[,]>();
        private Random rnd = new Random();
        private int length;

        public NeuralNetwork(params int[] layers)
        {
            this.layers = layers;
            SetWeights();
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double[,] Sigmoid(double[,] x)
        {
            int r = x.GetLength(0);
            int c = x.GetLength(1);
            double[,] y = new double[r, c];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    y[i, j] = Sigmoid(x[i, j]);

            return y;
        }

        private double _Sigmoid(double x)
        {
            return x * (1 - x);
        }

        private double[,] _Sigmoid(double[,] x)
        {
            int r = x.GetLength(0);
            int c = x.GetLength(1);
            double[,] y = new double[r, c];

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    y[i, j] = _Sigmoid(x[i, j]);

            return y;
        }

        private void SetWeights()
        {
            for (int i = 0; i < layers.Length - 1; i++)
            {
                double[,] weight = new double[layers[i], layers[i + 1]];
                int row = weight.GetLength(0);
                int col = weight.GetLength(1);
                for (int j = 0; j < row; j++)
                {
                    for (int k = 0; k < col; k++)
                    {
                        double d = rnd.NextDouble();
                        weight[j, k] = d;
                    }
                }
                weights.Add(weight);
                hiddens.Add(null);
                dHiddens.Add(null);
            }
            length = hiddens.Count - 1;
        }

        public void Update(double[,] x, double[,] y, int epochs, double learningRate)
        {
            for (int e = 0; e < epochs; e++)
            {
                hiddens[0] = Sigmoid(Dot(x, weights[0]));
                for (int i = 0; i < length; i++)
                    hiddens[i + 1] = Sigmoid(Dot(hiddens[i], weights[i + 1]));
                double[,] E = ErrorRate(y, hiddens[length]);
                dHiddens[length] = Mul(E, _Sigmoid(hiddens[length]));
                for (int i = length - 1; i >= 0; i--)
                    dHiddens[i] = Mul(Dot(dHiddens[i + 1], T(weights[i + 1])), _Sigmoid(hiddens[i]));
                for (int i = length - 1; i >= 0; i--)
                    Add(weights[i + 1], Dot(T(hiddens[i]), dHiddens[i + 1]), learningRate);
                Add(weights[0], Dot(T(x), dHiddens[0]), learningRate);
            }
        }

        public void Predict(double[,] x)
        {
            hiddens[0] = Sigmoid(Dot(x, weights[0]));
            for (int i = 0; i < length; i++)
                hiddens[i + 1] = Sigmoid(Dot(hiddens[i], weights[i + 1]));
            Print(hiddens[length]);
        }

        private double[,] ErrorRate(double[,] y, double[,] p)
        {
            int r = y.GetLength(0);
            int c = y.GetLength(1);
            double[,] e = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                    e[i, j] = y[i, j] - p[i, j];
            }
            return e;
        }

        private double[,] T(double[,] m)
        {
            int r = m.GetLength(0);
            int c = m.GetLength(1);

            double[,] t = new double[c, r];
            for (int i = 0; i < c; i++)
                for (int j = 0; j < r; j++)
                    t[i,j] = m[j,i];

            return t;
        }

        private void Add(double[,] a, double[,] b, double learningRate)
        {
            int r = a.GetLength(0);
            int c = b.GetLength(1);

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                    a[i, j] += learningRate * b[i, j];
            }
        }

        private double[,] Dot(double[,] A, double[,] B)
        {
            int rA = A.GetLength(0);
            int cA = A.GetLength(1);
            int rB = B.GetLength(0);
            int cB = B.GetLength(1);
            double temp = 0;
            double[,] result = new double[rA, cB];

            for (int i = 0; i < rA; i++)
            {
                for (int j = 0; j < cB; j++)
                {
                    temp = 0;
                    for (int k = 0; k < cA; k++)
                        temp += A[i, k] * B[k, j];
                    result[i, j] = temp;
                }
            }
            return result;
        }

        private double[,] Mul(double[,] A, double[,] B)
        {
            int r = A.GetLength(0);
            int c = A.GetLength(1);
            double[,] result = new double[r, c];

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                    result[i, j] = A[i, j] * B[i, j];
            }
            return result;
        }

        private void Print(double[,] matrix)
        {
            int r = matrix.GetLength(0);
            int c = matrix.GetLength(1);

            for (int i = 0; i < r; i++)
            {
                for (int j = 0; j < c; j++)
                    Console.Write(matrix[i, j] + " ");
                Console.WriteLine();
            }
        }
    }
}
