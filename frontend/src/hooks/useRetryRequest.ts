import { useState, useEffect, useRef } from 'react';

interface RetryOptions<T> {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  startDelay?: number;
  onSuccess?: (data: T) => void;
  onError?: (err: Error) => void;
  enabled?: boolean;
}

export function useRetryRequest<T>(
  requestFn: () => Promise<T>,
  options: RetryOptions<T> = {}
) {
  const { enabled = true } = options;
  
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);
  
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<Error | null>(null);
  
  const mounted = useRef(true);
  const attemptRef = useRef(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  // Use a ref for the latest function to avoid dependency issues if it changes
  const requestFnRef = useRef(requestFn);
  requestFnRef.current = requestFn;

  useEffect(() => {
    mounted.current = true;
    
    if (!enabled) {
      setLoading(false);
      return;
    }

    setLoading(true);
    attemptRef.current = 0;
    setError(null);

    const execute = async () => {
      try {
        const result = await requestFnRef.current();
        if (!mounted.current) return;
        setData(result);
        setLoading(false);
        if (optionsRef.current.onSuccess) optionsRef.current.onSuccess(result);
      } catch (err: any) {
        if (!mounted.current) return;
        
        const currentMaxRetries = optionsRef.current.maxRetries ?? 5;
        const currentInitialDelay = optionsRef.current.initialDelay ?? 500;
        const currentMaxDelay = optionsRef.current.maxDelay ?? 5000;

        if (attemptRef.current < currentMaxRetries) {
          const delay = Math.min(currentInitialDelay * Math.pow(2, attemptRef.current), currentMaxDelay);
          if (import.meta.env?.DEV) {
            console.log(`[useRetryRequest] Request failed, retrying (${attemptRef.current + 1}/${currentMaxRetries}) in ${delay}ms...`);
          }
          attemptRef.current++;
          timeoutRef.current = setTimeout(execute, delay);
        } else {
          setLoading(false);
          setError(err as Error);
          if (optionsRef.current.onError) optionsRef.current.onError(err as Error);
        }
      }
    };
    
    if (optionsRef.current.startDelay) {
      timeoutRef.current = setTimeout(execute, optionsRef.current.startDelay);
    } else {
      execute();
    }

    return () => {
      mounted.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [enabled]); // Re-run if enabled changes

  return { data, loading, error };
}
