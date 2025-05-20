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
        Console.WriteLine($"\n✅ Loaded data for {grouped.Count()} unique {itemLabel}(s).");

        var mlContext = new MLContext();
        int forecastedCount = 0;
        var allForecasts = new List<object>();

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

                forecastedCount++;
                Console.WriteLine($"\n📊 Forecast for {itemLabel} {name}:");

                var lastKnown = dataList.TakeLast(3).ToList();
                Console.WriteLine("🗂 Recent Actual Values:");
                foreach (var record in lastKnown)
                {
                    Console.WriteLine($"  {record.Date:yyyy-MM}: {record.Value} {(choice == "1" ? "$" : "plays")}");
                }

                Console.WriteLine("🔮 Predicted Next 3 Months:");
                var predictions = new List<(string Month, float Value)>();
                for (int i = 0; i < forecast.ForecastedValue!.Length; i++)
                {
                    var futureDate = dataList.Last().Date.AddMonths(i + 1);
                    float predictedValue = forecast.ForecastedValue[i];
                    Console.WriteLine($"  {futureDate:yyyy-MM}: {predictedValue:F0} {(choice == "1" ? "$" : "plays")}");
                    predictions.Add((futureDate.ToString("yyyy-MM"), predictedValue));
                }

                var result = new
                {
                    Name = name,
                    Category = itemLabel,
                    LastKnown = lastKnown.Select(r => new { Date = r.Date.ToString("yyyy-MM"), Value = r.Value }),
                    Forecast = predictions.Select(p => new { Month = p.Month, PredictedValue = p.Value })
                };

                allForecasts.Add(result);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Error forecasting {name}: {ex.Message}");
            }
        }

        if (forecastedCount == 0)
        {
            Console.WriteLine("\n⚠️ No forecast was generated. All entries were skipped (probably not enough data).");
        }
        else
        {
            Console.WriteLine($"\n✅ Forecasts completed for {forecastedCount} {itemLabel}(s).");

            string outputFolder = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\forecast"));
            Directory.CreateDirectory(outputFolder);

            string fileName = (choice == "1") ? "country_forecast.json" : "genre_forecast.json";
            string combinedOutputPath = Path.Combine(outputFolder, fileName);

            var jsonOptions = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
            string jsonContent = System.Text.Json.JsonSerializer.Serialize(allForecasts, jsonOptions);
            File.WriteAllText(combinedOutputPath, jsonContent);

            Console.WriteLine($"📁 All forecasts saved to: {combinedOutputPath}");
        }

        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
}
