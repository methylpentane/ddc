# filter_json.pyを実行する
# 学習の実行に不要なデータ(challengeより上の難易度、地雷ノーツ、"0000")を削る
# あと、permutationの項目でデータセットの水増しを設定している
source smd_0_push.sh

python3 filter_json.py \
	${SMDATA_DIR}/json_raw/${1} \
	${SMDATA_DIR}/json_filt${2}/${1} \
	--chart_types=dance-single \
	--chart_difficulties=Beginner,Easy,Medium,Hard,Challenge \
	--min_chart_feet=1 \
	--max_chart_feet=-1 \
	--substitutions=M,0,4,2 \
	--arrow_types=1,2,3 \
	--max_jump_size=-1 \
	--remove_zeros \
	--permutations=0123,3120,0213,3210
