using System;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace ParallelHeptics.FrontendUnity
{
    /// <summary>
    /// Sends ExperimentControl JSON to the existing Python backend over TCP.
    /// Connection, reconnection, and socket writes stay off the main thread so
    /// Play Mode remains responsive even when the backend is not running yet.
    /// </summary>
    public sealed class ExperimentControlClient : MonoBehaviour
    {
        private const string ToggleInteractionJson = "{\"moderatorAction\":\"toggle_interaction\"}";

        [SerializeField] private string backendHost = "localhost";
        [SerializeField] private int backendPort = 12344;
        [SerializeField] private int reconnectDelayMs = 500;
        [SerializeField] private bool connectOnStart = true;

        private readonly object _clientLock = new object();
        private readonly ConcurrentQueue<byte[]> _sendQueue = new ConcurrentQueue<byte[]>();
        private readonly AutoResetEvent _sendSignal = new AutoResetEvent(false);
        private TcpClient _client;
        private volatile bool _running;
        private Thread _workerThread;

        public bool IsConnected
        {
            get
            {
                lock (_clientLock)
                {
                    return _client != null && _client.Connected;
                }
            }
        }
        public bool IsDebug { get; set; } = true;

        public void Configure(string host, int port)
        {
            backendHost = host;
            backendPort = port;
        }

        private void OnEnable()
        {
            _running = true;
            _workerThread = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = "Experiment TCP Worker"
            };
            _workerThread.Start();
        }

        private void OnDisable()
        {
            _running = false;
            _sendSignal.Set();
            CloseClient();
            if (_workerThread != null && _workerThread.IsAlive && !_workerThread.Join(250) && IsDebug)
            {
                Debug.LogWarning("TCP worker thread did not stop within timeout; it will exit as a background thread.");
            }
        }

        public void SendQuestionInput(QuestionInput input)
        {
            string json = JsonUtility.ToJson(new ExperimentControl { questionInput = (int)input });
            EnqueueJson(json);
        }

        public void SendInteractionToggle()
        {
            EnqueueJson(ToggleInteractionJson);
        }

        private void EnqueueJson(string json)
        {
            _sendQueue.Enqueue(Encoding.UTF8.GetBytes(json + "\n"));
            _sendSignal.Set();
        }

        private void WorkerLoop()
        {
            if (connectOnStart)
            {
                EnsureConnected();
            }

            while (_running)
            {
                if (!_sendQueue.TryDequeue(out byte[] payload))
                {
                    _sendSignal.WaitOne(reconnectDelayMs);
                    continue;
                }

                while (_running && !SendPayloadFromWorker(payload))
                {
                    _sendSignal.WaitOne(reconnectDelayMs);
                }
            }

            while (_sendQueue.TryDequeue(out _))
            {
                // Drop unsent answers on shutdown rather than blocking Play Mode exit.
            }
        }

        private bool SendPayloadFromWorker(byte[] payload)
        {
            if (!EnsureConnected())
            {
                return false;
            }

            try
            {
                TcpClient client;
                lock (_clientLock)
                {
                    client = _client;
                }

                NetworkStream stream = client.GetStream();
                DrainAvailableResponses(stream);
                stream.Write(payload, 0, payload.Length);
                stream.Flush();
                DrainAvailableResponses(stream);
                if (IsDebug)
                {
                    Debug.Log($"Sent ExperimentControl: {Encoding.UTF8.GetString(payload)}");
                }

                return true;
            }
            catch (Exception ex)
            {
                if (IsDebug)
                {
                    Debug.LogWarning($"Failed to send ExperimentControl; will reconnect. {ex.Message}");
                }
                CloseClient();
                return false;
            }
        }

        private void DrainAvailableResponses(NetworkStream stream)
        {
            var buffer = new byte[1024];
            while (_running && stream.DataAvailable)
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead <= 0)
                {
                    return;
                }

                if (IsDebug)
                {
                    string response = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
                    if (!string.IsNullOrEmpty(response))
                    {
                        Debug.Log($"Backend TCP response: {response}");
                    }
                }
            }
        }

        private bool EnsureConnected()
        {
            while (_running && !IsConnected)
            {
                try
                {
                    var nextClient = new TcpClient { NoDelay = true };
                    nextClient.Connect(backendHost, backendPort);
                    lock (_clientLock)
                    {
                        _client?.Close();
                        _client = nextClient;
                    }

                    if (IsDebug)
                    {
                        Debug.Log($"Connected to backend TCP at {backendHost}:{backendPort}");
                    }
                    return true;
                }
                catch (Exception)
                {
                    Thread.Sleep(reconnectDelayMs);
                }
            }

            return IsConnected;
        }

        private void CloseClient()
        {
            lock (_clientLock)
            {
                try
                {
                    _client?.Close();
                }
                catch (Exception ex)
                {
                    if (IsDebug)
                    {
                        Debug.LogWarning($"TCP close warning: {ex.Message}");
                    }
                }
                finally
                {
                    _client = null;
                }
            }
        }
    }
}
