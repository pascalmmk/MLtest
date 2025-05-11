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
    public float PlayCount { get; set; }
}

public class TimeSeriesInput
{
    [LoadColumn(0)]
    public DateTime Date;

    [LoadColumn(1)]
    public float PlayCount;
}

public class ForecastOutput
{
    public float[]? ForecastedPlayCount { get; set; }
}

class Program
{
    static void Main()
    {
        Console.WriteLine("🎯 What would you like to forecast?");
        Console.WriteLine("1 - Artists");
        Console.WriteLine("2 - Albums");
        Console.WriteLine("3 - Tracks");
        Console.Write("Enter choice (1/2/3): ");
        string choice = Console.ReadLine()?.Trim() ?? "";

        string filePath;
        string itemLabel;

        switch (choice)
        {
            case "1":
                filePath = "top_artists_by_month.csv";
                itemLabel = "ArtistName";
                break;
            case "2":
                filePath = "top_albums_by_month.csv";
                itemLabel = "AlbumTitle";
                break;
            case "3":
                filePath = "top_tracks_by_month.csv";
                itemLabel = "TrackTitle";
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

                if (!float.TryParse(parts[2].Trim(), out var playCount))
                    return null;

                return new ForecastInput
                {
                    Name = parts[0].Trim(),
                    Date = date,
                    PlayCount = playCount
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
                PlayCount = r.PlayCount
            }).ToList();

            if (dataList.Count < 6)
            {
                Console.WriteLine($"⏭ Skipping {name} (only {dataList.Count} months of data)");
                continue;
            }

            var dataView = mlContext.Data.LoadFromEnumerable(dataList);
            var pipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(ForecastOutput.ForecastedPlayCount),
                inputColumnName: nameof(TimeSeriesInput.PlayCount),
                windowSize: 3,
                seriesLength: 6,
                trainSize: dataList.Count,
                horizon: 3);

            try
            {
                var model = pipeline.Fit(dataView);
                var forecastEngine = model.CreateTimeSeriesEngine<TimeSeriesInput, ForecastOutput>(mlContext);
                var forecast = forecastEngine.Predict();

                Console.WriteLine($"\n📊 Forecast for {name}:");
                for (int i = 0; i < forecast.ForecastedPlayCount!.Length; i++)
                {
                    Console.WriteLine($"Month {i + 1}: {forecast.ForecastedPlayCount[i]:F0} plays");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Error forecasting {name}: {ex.Message}");
            }
        }
    }
}
