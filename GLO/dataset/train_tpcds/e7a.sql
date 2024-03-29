SELECT MIN(mi.info) AS movie_budget,
MIN(mi_idx.info) AS movie_votes,
MIN(kt.kind) AS movie_type,
MIN(t.title) AS violent_movie
FROM cast_info ci, company_name cn, info_type it1, info_type it2, keyword k,
movie_companies mc, movie_info mi, movie_info_idx mi_idx, movie_keyword mk, title t, kind_type kt
WHERE ci.note in ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)')
AND cn.country_code = '[hk]'AND it1.info = 'genres' AND it2.info = 'votes'
AND k.keyword in ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital')
AND mc.note like '%(Blu-ray)%' AND mi.info in ('Horror', 'Thriller') AND t.production_year > 2000
AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = ci.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id
AND ci.movie_id = mi.movie_id AND ci.movie_id = mi_idx.movie_id AND ci.movie_id = mk.movie_id AND ci.movie_id = mc.movie_id
AND mi.movie_id = mi_idx.movie_id AND mi.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id
AND mi_idx.movie_id = mk.movie_id AND mi_idx.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id
AND it1.id = mi.info_type_id AND it2.id = mi_idx.info_type_id AND k.id = mk.keyword_id AND cn.id = mc.company_id AND t.kind_id =kt.id;
