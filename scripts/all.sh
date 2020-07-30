# "./all.sh [shell]" と実行すると、fraxtilとitgの両方で順番にシェルを実行
# trainに関してはgpu指定を追加したので使えない

for COLL in fraxtil itg
do
	echo "Executing ${1} for ${COLL}"
	${1} ${COLL}
	echo "--------------------------------------------"
done
