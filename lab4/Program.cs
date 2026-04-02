using System.Globalization;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace lab4;

public struct TrainTrip
{
    private readonly string trainNumber;
    private readonly float distance;
    private readonly float travelTime;
    private readonly float averageSpeed;

    public TrainTrip(string trainNumber, float distance, float travelTime, float averageSpeed)
    {
        this.trainNumber = trainNumber;
        this.distance = distance;
        this.travelTime = travelTime;
        this.averageSpeed = averageSpeed;
    }

    public string TrainNumber => trainNumber;
    public float Distance => distance;
    public float TravelTime => travelTime;
    public float AverageSpeed => averageSpeed;

    public float CalculatedSpeed => distance / travelTime;

    public TrainData ToTrainData() => new()
    {
        Distance = distance,
        TravelTime = travelTime,
        Label = averageSpeed,
    };
}

public class TrainData
{
    public float Distance { get; set; }
    public float TravelTime { get; set; }
    public float Label { get; set; }
}

public class TrainPrediction
{
    [ColumnName("Score")]
    public float Score { get; set; }
}

internal static class Program
{
    private static readonly MLContext MlContext = new(seed: 7);
    private static ITransformer? model;

    private static void Main()
    {
        List<TrainTrip> trips = BuildInitialTrips();

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("===== МЕНЮ =====");
            Console.WriteLine("1. Добавить запись");
            Console.WriteLine("2. Показать все записи");
            Console.WriteLine("3. Обучить модель");
            Console.WriteLine("4. Выполнить прогноз");
            Console.WriteLine("5. Выход");
            Console.Write("Выберите пункт (1-5): ");

            string choice = (Console.ReadLine() ?? string.Empty).Trim();
            Console.WriteLine();

            switch (choice)
            {
                case "1":
                    AddTrip(trips);
                    break;
                case "2":
                    PrintTrips(trips);
                    break;
                case "3":
                    TrainModel(trips);
                    break;
                case "4":
                    PredictSpeed();
                    break;
                case "5":
                    Console.WriteLine("Завершение работы.");
                    return;
                default:
                    Console.WriteLine("Некорректный выбор. Введите число от 1 до 5.");
                    break;
            }
        }
    }

    private static List<TrainTrip> BuildInitialTrips() =>
    [
        new("A-101", 120f, 1.7f, 70.6f),
        new("A-102", 120f, 2.0f, 60.0f),
        new("B-210", 200f, 2.5f, 80.0f),
        new("B-211", 200f, 3.0f, 66.7f),
        new("C-305", 350f, 4.0f, 87.5f),
        new("C-306", 350f, 4.8f, 72.9f),
        new("D-410", 480f, 5.5f, 87.3f),
        new("D-411", 480f, 6.2f, 77.4f),
        new("E-512", 620f, 7.0f, 88.6f),
        new("E-513", 620f, 8.0f, 77.5f),
        new("F-620", 760f, 8.5f, 89.4f),
        new("F-621", 760f, 9.5f, 80.0f),
    ];

    private static void AddTrip(List<TrainTrip> trips)
    {
        Console.WriteLine("Добавление новой записи");
        string trainNumber = ReadTrainNumber("Номер поезда: ");
        float distance = ReadPositiveFloat("Расстояние (км): ");
        float travelTime = ReadPositiveFloat("Время в пути (ч): ");
        float averageSpeed = ReadPositiveFloat("Средняя скорость (км/ч): ");

        float calculated = distance / travelTime;
        float relativeDifference = Math.Abs(averageSpeed - calculated) / calculated;
        if (relativeDifference > 0.15f)
        {
            Console.WriteLine(
                $"Предупреждение: введенная скорость заметно отличается от расчетной ({calculated:F2} км/ч).");
        }

        trips.Add(new TrainTrip(trainNumber, distance, travelTime, averageSpeed));
        Console.WriteLine("Запись добавлена.");
    }

    private static void PrintTrips(List<TrainTrip> trips)
    {
        if (trips.Count == 0)
        {
            Console.WriteLine("Список пуст.");
            return;
        }

        Console.WriteLine("Таблица поездок:");
        Console.WriteLine(
            $"{"№",2} | {"Поезд",-10} | {"Расстояние, км",13} | {"Время, ч",9} | {"Скорость, км/ч",14} | {"D/T, км/ч",10}");
        Console.WriteLine(new string('-', 76));

        for (int i = 0; i < trips.Count; i++)
        {
            TrainTrip trip = trips[i];
            Console.WriteLine(
                $"{i + 1,2} | {trip.TrainNumber,-10} | {trip.Distance,13:F1} | {trip.TravelTime,9:F2} | {trip.AverageSpeed,14:F2} | {trip.CalculatedSpeed,10:F2}");
        }
    }

    private static void TrainModel(List<TrainTrip> trips)
    {
        if (trips.Count < 2)
        {
            Console.WriteLine("Недостаточно данных для обучения.");
            return;
        }

        List<TrainData> trainData = trips.Select(t => t.ToTrainData()).ToList();
        IDataView dataView = MlContext.Data.LoadFromEnumerable(trainData);

        IEstimator<ITransformer> pipeline = MlContext.Transforms
            .Concatenate("Features", nameof(TrainData.Distance), nameof(TrainData.TravelTime))
            .Append(MlContext.Regression.Trainers.Sdca(
                labelColumnName: nameof(TrainData.Label),
                featureColumnName: "Features"));

        model = pipeline.Fit(dataView);
        Console.WriteLine("Модель обучена.");
    }

    private static void PredictSpeed()
    {
        if (model is null)
        {
            Console.WriteLine("Сначала обучите модель (пункт 3).");
            return;
        }

        Console.WriteLine("Выполнение прогноза");
        string trainNumber = ReadTrainNumber("Номер поезда: ");
        float distance = ReadPositiveFloat("Расстояние (км): ");
        float travelTime = ReadPositiveFloat("Время в пути (ч): ");

        TrainData input = new() { Distance = distance, TravelTime = travelTime };
        PredictionEngine<TrainData, TrainPrediction> engine =
            MlContext.Model.CreatePredictionEngine<TrainData, TrainPrediction>(model);

        TrainPrediction prediction = engine.Predict(input);
        float calculated = distance / travelTime;

        Console.WriteLine();
        Console.WriteLine("Результат прогноза:");
        Console.WriteLine($"Поезд: {trainNumber}");
        Console.WriteLine($"Расстояние: {distance:F1} км");
        Console.WriteLine($"Время в пути: {travelTime:F2} ч");
        Console.WriteLine($"Предсказанная средняя скорость: {prediction.Score:F2} км/ч");
        Console.WriteLine($"Расчетная скорость distance / travelTime: {calculated:F2} км/ч");
        Console.WriteLine($"Разница: {Math.Abs(prediction.Score - calculated):F2} км/ч");
    }

    private static float ReadPositiveFloat(string prompt)
    {
        while (true)
        {
            Console.Write(prompt);
            string raw = (Console.ReadLine() ?? string.Empty).Trim().Replace(',', '.');

            if (float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && value > 0)
            {
                return value;
            }

            Console.WriteLine("Ошибка: введите число больше 0.");
        }
    }

    private static string ReadTrainNumber(string prompt)
    {
        while (true)
        {
            Console.Write(prompt);
            string value = (Console.ReadLine() ?? string.Empty).Trim();

            if (Regex.IsMatch(value, "^[A-Za-zА-Яа-я0-9-]{2,10}$"))
            {
                return value;
            }

            Console.WriteLine("Ошибка: номер должен быть длиной 2-10 символов (буквы, цифры, дефис).");
        }
    }
}
