CREATE OR REPLACE FUNCTION public.match_documents(query_embedding vector, match_count integer DEFAULT 5, filter jsonb DEFAULT '{}'::jsonb)
 RETURNS TABLE(id bigint, doc_id text, chunk_index integer, content text, metadata jsonb, similarity double precision)
 LANGUAGE plpgsql
 STABLE
 SECURITY DEFINER
 SET search_path = public
AS $function$
begin
  return query
  select c.id, c.doc_id, c.chunk_index, c.content, c.metadata,
         1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks c
  where (filter = '{}'::jsonb) or (c.metadata @> filter)
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$function$;

GRANT EXECUTE ON FUNCTION public.match_documents(vector, integer, jsonb) TO anon, authenticated;