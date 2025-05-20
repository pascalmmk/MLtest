using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

public class ForecastInput
{
    public string? Name { get; set; }
    public DateTime Date { get; set; }
    public float Value { get; set; }
}

public class TimeSeriesInput
{
    [LoadColumn(0)]
    public DateTime Date;

    [LoadColumn(1)]
    public float Value;
}

public class ForecastOutput
{
    public float[]? ForecastedValue { get; set; }
}

class Program
{
    static void Main()
    {
        Console.WriteLine("🎯 What would you like to forecast?");
        Console.WriteLine("1 - Country Spending");
        Console.WriteLine("2 - Top Genres by Country listening data");
        Console.Write("Enter choice (1/2): ");
        string choice = Console.ReadLine()?.Trim() ?? "";

        string filePath;
        string itemLabel;

        switch (choice)
        {
            case "1":
                filePath = "top_country_spending_by_month_EXPANDED.csv";
                itemLabel = "Country";
                break;
            case "2":
                filePath = "top_genres_by_month_EXPANDED.csv";
                itemLabel = "Genre";
                break;
            default:
                Console.WriteLine("❌ Invalid choice. Exiting.");
                return;
        }

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"❌ File not found: {filePath}");
            return;
        }

        var lines = File.ReadAllLines(filePath).Skip(1);
        var allData = lines
            .Select(line =>
            {
                var parts = line.Split(',');
                if (parts.Length < 3) return null;

                if (!DateTime.TryParseExact(parts[1].Trim(), "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var date))
                    return null;

                if (!float.TryParse(parts[2].Trim(), out var value))
                    return null;

                return new ForecastInput
                {
                    Name = parts[0].Trim(),
                    Date = date,
                    Value = value
                };
            })
            .Where(r => r != null)
            .ToList()!;

        var grouped = allData.GroupBy(r => r.Name);

        var mlContext = new MLContext();

        foreach (var group in grouped)
        {
            var name = group.Key!;
            var dataList = group.OrderBy(r => r.Date).Select(r => new TimeSeriesInput
            {
                Date = r.Date,
                Value = r.Value
            }).ToList();

            if (dataList.Count < 6)
            {
                Console.WriteLine($"⏭ Skipping {name} (only {dataList.Count} months of data)");
                continue;
            }

            var dataView = mlContext.Data.LoadFromEnumerable(dataList);
            var pipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(ForecastOutput.ForecastedValue),
                inputColumnName: nameof(TimeSeriesInput.Value),
                windowSize: 3,
                seriesLength: 6,
                trainSize: dataList.Count,
                horizon: 3);

            try
            {
                var model = pipeline.Fit(dataView);
                var forecastEngine = model.CreateTimeSeriesEngine<TimeSeriesInput, ForecastOutput>(mlContext);
                var forecast = forecastEngine.Predict();

                Console.WriteLine($"\n📊 Forecast for {itemLabel} {name}:");

                var lastKnown = dataList.TakeLast(3).ToList();
                Console.WriteLine("🗂 Recent Actual Values:");
                foreach (var record in lastKnown)
                {
                    Console.WriteLine($"  {record.Date:yyyy-MM}: {record.Value} {(choice == "1" ? "$" : "plays")}");
                }

                Console.WriteLine("🔮 Predicted Next 3 Months:");
                for (int i = 0; i < forecast.ForecastedValue!.Length; i++)
                {
                    var futureDate = dataList.Last().Date.AddMonths(i + 1);
                    Console.WriteLine($"  {futureDate:yyyy-MM}: {forecast.ForecastedValue[i]:F0} {(choice == "1" ? "$" : "plays")}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Error forecasting {name}: {ex.Message}");
            }
        }
    }
}
