using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(2,4,4,1);
            double[,] x = new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            double[,] y = new double[,] { { 0.5 }, { 1 }, { 0.5 }, { 0 } };
            nn.Update(x, y, 5000, 1);
            nn.Predict(x);
            Console.ReadLine();
        }
    }
}
