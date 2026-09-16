declare module 'react-plotly.js' {
  import * as React from 'react';
  import { Layout, Config, Data } from 'plotly.js';

  export interface PlotParams {
    data: Data[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    frames?: any[];
    useResizeHandler?: boolean;
    style?: React.CSSProperties;
    className?: string;
    onInitialized?: (figure: Readonly<any>, graphDiv: Readonly<HTMLElement>) => void;
    onUpdate?: (figure: Readonly<any>, graphDiv: Readonly<HTMLElement>) => void;
    onPurge?: (figure: Readonly<any>, graphDiv: Readonly<HTMLElement>) => void;
    onError?: (err: Error) => void;
    divId?: string;
  }

  export default class Plot extends React.Component<PlotParams> {}
}
