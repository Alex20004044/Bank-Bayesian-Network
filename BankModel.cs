using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using System.Security.Cryptography.X509Certificates;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace IS_1_PGMBank
{
    class BankModel
    {
        Variable<bool> isLawful;
        Variable<bool> isYoung;
        /// <summary>
        /// Ratio of Debts To Income
        /// </summary>
        Variable<bool> isGoodCreditRate;
        Variable<bool> isGoodPaymentHistory;
        Variable<bool> isReliable;

        Variable<bool> isGoodEducation;
        Variable<bool> isHighIncome;
        Variable<bool> isManyAssets;
        Variable<bool> isBigFutureIncome;

        Variable<bool> isHasDualCitizenship;

        Variable<bool> isGoodCreditWorthiness;

        public void Run()
        {
            CreateModel();
            DefineObservation();
            Compute();
        }

        void CreateModel()
        {
            isGoodEducation = Variable.Bernoulli(0.3).Named(nameof(isGoodEducation));

            isHighIncome = Variable.New<bool>().Named(nameof(isHighIncome));
            using (Variable.If(isGoodEducation)) isHighIncome.SetTo(Variable.Bernoulli(0.4));
            using (Variable.IfNot(isGoodEducation)) isHighIncome.SetTo(Variable.Bernoulli(0.2));


            isManyAssets = Variable.New<bool>().Named(nameof(isManyAssets));
            using (Variable.If(isHighIncome)) isManyAssets.SetTo(Variable.Bernoulli(0.9));
            using (Variable.IfNot(isHighIncome)) isManyAssets.SetTo(Variable.Bernoulli(0.2));

            isBigFutureIncome = Variable.New<bool>().Named(nameof(isBigFutureIncome));
            CreateDependency(isHighIncome, isManyAssets, isBigFutureIncome,
                .95, .9, .7, .3);

            isHasDualCitizenship = Variable.Bernoulli(0.05);

            isYoung = Variable.Bernoulli(0.6);

            isGoodCreditRate = Variable.Bernoulli(0.4);

            isGoodPaymentHistory = Variable.New<bool>();
            CreateDependency(isYoung, isGoodCreditRate, isGoodPaymentHistory,
                .8,.6,.95,.7);

            isLawful = Variable.Bernoulli(0.9);

            isReliable = Variable.New<bool>();
            CreateDependency(isGoodPaymentHistory, isYoung, isLawful, isReliable,
                .7, .6, .9, .8,
                .6, .5, .7, .6);
            isGoodCreditWorthiness = false;
            isGoodCreditWorthiness = Variable.New<bool>();
            /*            using (Variable.If(isHasDualCitizenship))
                        {
                            CreateDependency(isReliable, isBigFutureIncome, isGoodCreditRate, isGoodCreditWorthiness,
                                .98, .85, .35, .25,
                                .64, .25, .5, 0.005);
                        }
                        using (Variable.IfNot(isHasDualCitizenship))
                        {
                            CreateDependency(isReliable, isBigFutureIncome, isGoodCreditRate, isGoodCreditWorthiness,
                                .99, .9, .4, .3,
                                .75, .3, .5, 0.01);
                        }*/
            using (Variable.If(isReliable))
            {
                CreateDependency(isBigFutureIncome, isGoodCreditRate, isHasDualCitizenship, isGoodCreditWorthiness,
                    .99, .98, .9, .85,
                    .4, .35, .3, 0.25);
            }
            using (Variable.IfNot(isReliable))
            {
                CreateDependency(isBigFutureIncome, isGoodCreditRate, isHasDualCitizenship, isGoodCreditWorthiness,
                    .75, .64, .3, .25,
                    .5, .5, .01, 0.005);
            }
        }

        private void CreateDependency(Variable<bool> parentA, Variable<bool> parentB, Variable<bool> parentC, Variable<bool> dependent,
            double ABC, double ABnotC, double AnotBC, double AnotBnotC, 
            double notABC, double notABnotC, double notAnotBC, double notAnotBnotC)
        {
            using (Variable.If(parentA))
            {
                CreateDependency(parentB, parentC, dependent,
                    ABC, ABnotC, AnotBC, AnotBnotC);
            }
            using (Variable.IfNot(parentA))
            {
                CreateDependency(parentB, parentC, dependent,
                    notABC, notABnotC, notAnotBC, notAnotBnotC);
            }
        }

        private void CreateDependency(Variable<bool> parentA, Variable<bool> parentB, Variable<bool> dependent,
            double AB, double AnotB, double notAB, double notAnotB)
        {
            using (Variable.If(parentA))
            {
                using (Variable.If(parentB))
                    dependent.SetTo(Variable.Bernoulli(AB));
                using (Variable.IfNot(parentB))
                    dependent.SetTo(Variable.Bernoulli(AnotB));
            }
            using (Variable.IfNot(parentA))
            {
                using (Variable.If(parentB))
                    dependent.SetTo(Variable.Bernoulli(notAB));
                using (Variable.IfNot(parentB))
                    dependent.SetTo(Variable.Bernoulli(notAnotB));
            }
        }

        void Compute()
        {
            InferenceEngine engine = new InferenceEngine();

            Console.WriteLine($"{nameof(isGoodEducation)}: " + engine.Infer(isGoodEducation));
            Console.WriteLine($"{nameof(isHighIncome)}: " + engine.Infer(isHighIncome));
            Console.WriteLine($"{nameof(isManyAssets)}: " + engine.Infer(isManyAssets));
            Console.WriteLine($"{nameof(isBigFutureIncome)}: " + engine.Infer(isBigFutureIncome));
            Console.WriteLine($"{nameof(isHasDualCitizenship)}: " + engine.Infer(isHasDualCitizenship));
            Console.WriteLine($"{nameof(isYoung)}: " + engine.Infer(isYoung));
            Console.WriteLine($"{nameof(isGoodCreditRate)}: " + engine.Infer(isGoodCreditRate));
            Console.WriteLine($"{nameof(isGoodPaymentHistory)}: " + engine.Infer(isGoodPaymentHistory));
            Console.WriteLine($"{nameof(isLawful)}: " + engine.Infer(isLawful));
            Console.WriteLine($"{nameof(isReliable)}: " + engine.Infer(isReliable));
            Console.WriteLine($"{nameof(isGoodCreditWorthiness)}: " + engine.Infer(isGoodCreditWorthiness));
        }

        void DefineObservation()
        {
            //isGoodEducation.ObservedValue = true;
            //isHighIncome.ObservedValue = true;
            //isManyAssets.ObservedValue = true;
            //isBigFutureIncome.ObservedValue = true;
            //isHasDualCitizenship.ObservedValue = true;
            //isYoung.ObservedValue = true;
            //isGoodCreditRate.ObservedValue = false;
            //isGoodPaymentHistory.ObservedValue = false;
            //isLawful.ObservedValue = false;
            //isReliable.ObservedValue = true;
            //isGoodCreditWorthiness.ObservedValue = false;
        }


    }
}
