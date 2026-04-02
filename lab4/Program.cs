using System;

namespace lab4;

public struct TrainTrip
{
    private string trainNumber;
    private float distance;
    private float travelTime;
    private float averageSpeed;

    public TrainTrip(string trainNumber, float distance, float travelTime, float averageSpeed)
    {
        this.trainNumber = trainNumber;
        this.distance = distance;
        this.travelTime = travelTime;
        this.averageSpeed = averageSpeed;
    }

    public void Print()
    {
        Console.WriteLine(
            $"Поезд: {trainNumber,-8} | Расстояние: {distance,7:F1} км | Время: {travelTime,6:F2} ч | Средняя скорость: {averageSpeed,7:F2} км/ч");
    }

    public TrainData ToMlData()
    {
        return new TrainData
        {
            Distance = distance,
            TravelTime = travelTime,
            Label = averageSpeed,
        };
    }

    public string TrainNumber => trainNumber;
    public float Distance => distance;
    public float TravelTime => travelTime;
    public float AverageSpeed => averageSpeed;
}

public class TrainData
{
    public float Distance { get; set; }
    public float TravelTime { get; set; }
    public float Label { get; set; }
}

public class TrainPrediction
{
    public float PredictedAverageSpeed { get; set; }
}

internal static class Program
{
    private static void Main()
    {
        Console.WriteLine("Структура TrainTrip и ML-классы добавлены в проект.");
    }
}
