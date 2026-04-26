using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace ParallelHeptics.FrontendUnity
{
    /// <summary>
    /// Receives ExperimentPacket JSON without blocking Unity's main thread.
    /// Stores only the newest datagram so fast backend bursts cannot create a
    /// stale-packet backlog.
    /// </summary>
    public sealed class ExperimentUdpReceiver : MonoBehaviour
    {
        [SerializeField] private int listenPort = 12346;
        [SerializeField] private bool bindLoopbackOnly;
        [SerializeField] private int receiveBufferBytes = 8192;
        [SerializeField] private bool logPacketsPerSecond;

        private readonly object _latestJsonLock = new object();
        private UdpClient _udpClient;
        private Thread _receiveThread;
        private volatile bool _running;
        private string _latestJson;
        private uint _latestSequence;
        private uint _lastConsumedSequence;
        private long _receivedPackets;
        private float _lastStatsTime;
        private long _lastStatsPackets;

        public int ListenPort => listenPort;
        public long ReceivedPackets => Interlocked.Read(ref _receivedPackets);
        public bool IsRunning => _running;

        public void Configure(int port, bool loopbackOnly)
        {
            if (_running && port != listenPort)
            {
                Debug.LogWarning("UDP port changed while receiver is running; restart Play Mode to rebind.");
            }

            listenPort = port;
            bindLoopbackOnly = loopbackOnly;
        }

        private void OnEnable()
        {
            StartReceiver();
        }

        private void OnDisable()
        {
            StopReceiver();
        }

        private void Update()
        {
            if (!logPacketsPerSecond || Time.unscaledTime - _lastStatsTime < 1f)
            {
                return;
            }

            long packets = ReceivedPackets;
            long delta = packets - _lastStatsPackets;
            _lastStatsPackets = packets;
            _lastStatsTime = Time.unscaledTime;
            Debug.Log($"Unity frontend UDP receive rate: {delta} packets/s on port {listenPort}");
        }

        public bool TryGetLatestJson(out string json)
        {
            lock (_latestJsonLock)
            {
                if (_latestSequence == _lastConsumedSequence || string.IsNullOrEmpty(_latestJson))
                {
                    json = null;
                    return false;
                }

                _lastConsumedSequence = _latestSequence;
                json = _latestJson;
                return true;
            }
        }

        private void StartReceiver()
        {
            if (_running)
            {
                return;
            }

            try
            {
                IPAddress bindAddress = bindLoopbackOnly ? IPAddress.Loopback : IPAddress.Any;
                _udpClient = new UdpClient(new IPEndPoint(bindAddress, listenPort));
                _udpClient.Client.ReceiveBufferSize = receiveBufferBytes;
                _running = true;
                _receiveThread = new Thread(ReceiveLoop)
                {
                    IsBackground = true,
                    Name = "Experiment UDP Receiver"
                };
                _receiveThread.Start();
                Debug.Log($"Experiment UDP receiver listening on {(bindLoopbackOnly ? "127.0.0.1" : "0.0.0.0")}:{listenPort}");
            }
            catch (Exception ex)
            {
                _running = false;
                Debug.LogError($"Failed to start Experiment UDP receiver on port {listenPort}: {ex.Message}");
            }
        }

        private void StopReceiver()
        {
            _running = false;
            try
            {
                _udpClient?.Close();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"UDP close warning: {ex.Message}");
            }

            if (_receiveThread != null && _receiveThread.IsAlive && !_receiveThread.Join(250))
            {
                Debug.LogWarning("UDP receiver thread did not stop within timeout; it will exit as a background thread.");
            }

            _receiveThread = null;
            _udpClient = null;
        }

        private void ReceiveLoop()
        {
            IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    byte[] payload = _udpClient.Receive(ref remoteEndPoint);
                    string json = Encoding.UTF8.GetString(payload);
                    lock (_latestJsonLock)
                    {
                        _latestJson = json;
                        unchecked
                        {
                            _latestSequence++;
                        }
                    }

                    Interlocked.Increment(ref _receivedPackets);
                }
                catch (SocketException)
                {
                    if (_running)
                    {
                        Debug.LogWarning("UDP receive socket exception; receiver will continue if still running.");
                    }
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (Exception ex)
                {
                    if (_running)
                    {
                        Debug.LogWarning($"UDP receive warning: {ex.Message}");
                    }
                }
            }
        }
    }
}
