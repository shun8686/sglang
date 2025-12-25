LOGPATH="./lts_test_log/`date  +"%y%m%d%H%M%S"`"

function sglangMonitor()
{
	while true
	do
		sglangPid=$(ps -ef |grep "python3 -m sglang.launch_server" |awk '{print $2}'|head -1)
		sglangLsopOpenFile=$(lsof -p $sglangPid|wc -l)
		sglangRES=$(top -bn1 -p ${sglangPid} |tail -n2|grep ${sglangPid}|awk '{print $6}')
		sglangMEM=$(top -bn1 -p ${sglangPid} |tail -n2|grep ${sglangPid}|awk '{print $10}')
		sglangCPU=$(top -bn1 -p ${sglangPid} |tail -n2|grep ${sglangPid}|awk '{print $9}')
		sglangZoom=$(ps -ef |grep defunc[t]|wc -l)
		echo "`date +"%y%m%d-%H:%M:%S"` sglangPid:${sglangPid} sglangCPU:"${sglangCPU}%" sglangRES:${sglangRES} sglangMEM:"${sglangMEM}%" sglangLsopOpenFile:${sglangLsopOpenFile} sglangZoom:${sglangZoom} " >> $LOGPATH/server_log.csv
		sleep 10
        done

}

function nodeMonitor()
{

	while true
	do
		nodeSYCPU=$(top -bn1 |grep Cpu |awk '{print $4}')
		nodeUSCPU=$(top -bn1 |grep Cpu |awk '{print $2}')
		nodeCPU=$(echo ${nodeSYCPU} + ${nodeUSCPU} | bc)
		nodemem_kb=$(vmstat -s |grep "used memory"|awk '{print $1}')
		nodemem=$(awk "BEGIN {print $nodemem_kb/1024/1024}")
		echo "`date  +"%y%m%d-%H:%M:%S"` nodeSYCPU:"${nodeSYCPU}%" nodeUSCPU:"${nodeUSCPU}%" nodeCPU:"${nodeCPU}%" nodemem:"${nodemem}g" " >> $LOGPATH/node_log.csv
		sleep 10
       done
}

[[ ! -d ${LOGPATH} ]] && mkdir -p ${LOGPATH}

[[ -z $1 ]] && exit 1
case $1 in
	server)
		sglangMonitor
	;;
        node)
		nodeMonitor
	;;
	*)
		exit 1
	;;
esac
