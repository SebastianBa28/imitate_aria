# Sourced by pixi on activation (pixi.toml [activation] scripts).
# ROBOT_NET_IP: this machine's address on the robot network (where the Franka NUC is), used by
# config/fastdds.xml. With several network interfaces (wired + Wi-Fi + docker), Fast DDS otherwise
# announces all of them and the controller's data never arrives. Off that network: loopback only.
_net="${ROBOT_NET_PREFIX:-172.16.0.}"
if command -v ip >/dev/null 2>&1; then
    ROBOT_NET_IP="$(ip -4 -o addr show 2>/dev/null | awk -v p="$_net" 'index($4, p) == 1 {split($4, a, "/"); print a[1]; exit}')"
else  # macOS
    ROBOT_NET_IP="$(ifconfig 2>/dev/null | awk -v p="$_net" '$1 == "inet" && index($2, p) == 1 {print $2; exit}')"
fi
export ROBOT_NET_IP="${ROBOT_NET_IP:-127.0.0.1}"
unset _net
